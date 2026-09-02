import numpy as np
from PIL import Image

from scripts.prep_perturbations import centre_scale, directional_blur, occlude
from scripts.perturbation_extract import parser as extraction_parser
from scripts.perturbation_run import condition_details


def patterned_image() -> Image.Image:
    values = np.zeros((20, 30, 3), dtype=np.uint8)
    values[:, 15:] = 255
    return Image.fromarray(values)


def test_centre_scale_keeps_canvas_and_records_crop():
    transformed, parameters = centre_scale(patterned_image(), 2.0)

    assert transformed.size == (30, 20)
    assert parameters["apparent_scale"] == 2.0
    assert parameters["crop_fraction"] == 0.25


def test_occlusion_records_the_actual_rectangle_area():
    image = patterned_image()
    transformed, parameters = occlude(image, 0.25, np.random.default_rng(0))

    assert transformed.size == image.size
    assert parameters["actual_area_fraction"] == 0.25
    assert parameters["fill_rgb"] == [127, 127, 127]


def test_directional_blur_keeps_shape_and_changes_an_edge():
    image = patterned_image()
    transformed = directional_blur(image, radius=4, angle_degrees=0.0)

    assert transformed.size == image.size
    assert not np.array_equal(np.asarray(transformed), np.asarray(image))


def test_condition_details_maps_all_recorded_levels():
    manifest = {
        "factors": {
            "scale": [1.0, 2.0],
            "occlusion": [0.0, 0.5],
            "motion_blur_radius_pixels": [0, 4],
        }
    }

    details = condition_details(manifest)

    assert details["scale_l1"] == {"factor": "scale", "level": 1, "value": 2.0}
    assert details["occlusion_l1"]["value"] == 0.5
    assert details["motion_blur_l1"]["value"] == 4


def test_condition_details_accepts_budget_reduced_single_factor():
    details = condition_details({"factors": {"occlusion": [0.0, 0.1, 0.5]}})

    assert set(details) == {"identity", "occlusion_l1", "occlusion_l2"}


def test_perturbation_extraction_defaults_to_fixed_2000_images():
    args = extraction_parser().parse_args([
        "--model", "dinov2", "--identity-images", "clean", "--condition-images", "hard",
        "--transform-manifest", "manifest.json", "--out", "cache", "--batch-size", "16",
    ])

    assert args.count == 2_000
