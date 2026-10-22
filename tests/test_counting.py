import numpy as np
import pytest
import torch

from vtb import counting


def test_centre_crop_frame_uses_integer_offsets_and_half_open_edges():
    frame = counting.centre_crop_frame(width=577, height=384)
    assert (frame.left, frame.top, frame.side) == (96, 0, 384)
    points = frame.points(np.array([[96.0, 0.0], [479.999, 383.9], [480.0, 10.0], [95.99, 10.0]]))
    assert counting.inside(points).tolist() == [True, True, False, False]
    assert frame.valid == (0.0, 0.0, 1.0, 1.0)


def test_letterbox_frame_keeps_every_point_and_reports_valid_area():
    frame = counting.letterbox_frame(width=600, height=400)
    assert (frame.left, frame.top, frame.side) == (0, -100, 600)
    corners = frame.points(np.array([[0.0, 0.0], [599.99, 399.99]]))
    assert counting.inside(corners).all()
    assert frame.valid == pytest.approx((0.0, 100 / 600, 1.0, 500 / 600))
    cover = counting.coverage(frame.valid, 6)
    assert cover.sum() == pytest.approx(6 * 6 * 400 / 600)
    assert cover[0].sum() == 0 and cover[2].tolist() == [1.0] * 6


@pytest.mark.parametrize("grid", [64, 17])
def test_density_conserves_mass_at_edges_and_corners(grid):
    rng = np.random.default_rng(0)
    points = np.concatenate([rng.random((50, 2)), [[0.0, 0.0], [0.9999, 0.9999], [0.0, 0.5]]])
    density = counting.density_map(points, grid=grid)
    assert density.shape == (grid, grid) and (density >= 0).all()
    assert density.sum(dtype=np.float64) == pytest.approx(len(points), abs=1e-4)
    single = counting.density_map(np.array([[0.0, 0.0]]), grid=grid)
    assert single.sum(dtype=np.float64) == pytest.approx(1.0, abs=1e-6)
    assert single[0, 0] == single.max()


def test_density_handles_zero_count_and_rejects_dropped_points():
    assert counting.density_map(np.zeros((0, 2))).sum() == 0
    with pytest.raises(ValueError):
        counting.density_map(np.array([[1.0, 0.5]]))


def test_clip_box_reports_the_kept_area():
    clipped, kept = counting.clip_box(np.array([-0.1, 0.2, 0.1, 0.4]))
    assert clipped.tolist() == pytest.approx([0.0, 0.2, 0.1, 0.4])
    assert kept == pytest.approx(0.5)


def test_box_weights_follow_area_overlap():
    weights = counting.box_weights(np.array([[0.0, 0.0, 0.5, 0.25]]), side=4)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0, 0, :2].tolist() == pytest.approx([0.5, 0.5])
    assert weights[0, 1:].sum() == 0 and weights[0, :, 2:].sum() == 0


def test_prototype_similarity_marks_the_exemplar_pattern_with_any_box_count():
    features = torch.zeros(1, 3, 4, 4)
    features[0, 0] = 1.0
    features[0, :, 0, 0] = torch.tensor([0.0, 5.0, 0.0])
    features[0, :, 3, 3] = torch.tensor([0.0, 5.0, 0.0])
    one = counting.prototype_similarity(features, [np.array([[0.0, 0.0, 0.25, 0.25]])])
    three = counting.prototype_similarity(
        features, [np.array([[0.0, 0.0, 0.25, 0.25], [0.75, 0.75, 1.0, 1.0], [0.0, 0.0, 0.25, 0.25]])]
    )
    for similarity in (one, three):
        assert similarity.shape == (1, 1, 4, 4)
        assert similarity[0, 0, 3, 3] == pytest.approx(1.0, abs=1e-5)
        assert similarity[0, 0, 1, 1] < 0
    assert counting.condition(features, [np.array([[0, 0, 0.25, 0.25]])]).shape == (1, 4, 4, 4)
    with pytest.raises(ValueError):
        counting.prototype_similarity(features, [np.zeros((0, 4))])


def test_masked_count_weights_cells_by_valid_area():
    density = torch.ones(2, 1, 4, 4)
    valid = torch.from_numpy(counting.coverage((0.0, 0.0, 1.0, 0.5), 4)).expand(2, 4, 4)
    assert counting.masked_count(density).tolist() == [16.0, 16.0]
    assert counting.masked_count(density, valid).tolist() == [8.0, 8.0]


def test_count_bins_use_inclusive_upper_edges():
    assert [counting.count_bin(c) for c in (0, 50, 50.5, 200, 201)] == [
        "le50", "le50", "50to200", "50to200", "gt200",
    ]


def test_dots_on_the_closing_image_edge_are_clamped_onto_the_image():
    dots, moved = counting.onto_image(np.array([[384.0, 10.0], [5.0, 384.0], [383.5, 2.0], [390.0, 3.0]]), 384, 384)
    assert moved == 2
    frame = counting.centre_crop_frame(384, 384)
    assert counting.inside(frame.points(dots)).tolist() == [True, True, True, False]


@pytest.mark.parametrize("source,extent", [(256, 1.0), (224, 1.0), (112, 1.0), (240, 30 / 28), (88, 462 / 448)])
def test_area_matrix_averages_the_canvas_part_of_the_grid(source, extent):
    weights = counting.area_matrix(source, 64, extent)
    assert weights.shape == (64, source) and np.allclose(weights.sum(axis=1), 1.0)
    # Source pixels beyond the canvas (padding) never feed a target cell.
    beyond = np.arange(source) * extent / source >= 1.0
    assert weights[:, beyond].sum() == 0
    if extent == 1.0 and source % 64 == 0:
        step = source // 64
        assert np.allclose(weights[3, 3 * step:4 * step], 1 / step)


def test_padded_grids_map_boxes_and_validity_through_the_extent():
    extent = 30 / 28
    weights = counting.box_weights(np.array([[0.0, 0.0, 1.0, 1.0]]), 30, extent)
    assert weights[0, :28, :28].sum() == pytest.approx(1.0) and weights[0, 28:].sum() == 0
    valid = counting.token_valid((0.0, 0.0, 1.0, 1.0), 30, extent)
    assert valid[:28, :28].min() == 1.0 and valid[28:].max() == 0.0
