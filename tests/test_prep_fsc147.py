import json
import sys

import numpy as np
import pytest
from PIL import Image

from scripts import prep_fsc147 as prep
from vtb import counting


def corners(x0, y0, x1, y1):
    """Release layout: four corners, starting top-left and running counter-clockwise like the release."""
    return [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]


def write_release(root, entries):
    """entries: name -> (split, class, (w, h), points, boxes as corner lists)."""
    raw, images = root / "raw", root / "images"
    raw.mkdir(parents=True)
    images.mkdir()
    rng = np.random.default_rng(0)
    annotations, splits, lines = {}, {"train": [], "val": [], "test": [], "val_coco": []}, []
    for name, (split, label, (w, h), points, boxes) in entries.items():
        pixels = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        # Lossless bytes under the release's .jpg name, so crops can be compared exactly.
        Image.fromarray(pixels).save(images / name, format="PNG")
        annotations[name] = {"H": h * 2, "W": w * 2, "ratio_h": 0.5, "ratio_w": 0.5,
                             "points": points, "box_examples_coordinates": boxes}
        splits[split].append(name)
        lines.append(f"{name}\t{label}")
    (raw / prep.ANNOTATIONS).write_text(json.dumps(annotations))
    (raw / prep.SPLIT_FILE).write_text(json.dumps(splits))
    (raw / prep.CLASS_FILE).write_text("\n".join(lines) + "\n")
    return raw, images


THREE = [corners(3, 1, 4, 2), corners(4, 2, 5, 3), corners(5, 3, 6, 4)]


def release(root):
    return write_release(root, {
        # 10 x 6: the crop keeps columns [2, 8).
        "1.jpg": ("train", "apples", (10, 6),
                  [[2.0, 0.0], [7.999, 5.999], [8.0, 3.0], [1.999, 3.0], [5.0, 6.0], [4.5, 2.5]], THREE),
        # Every dot outside the crop: a zero-count image that stays in the set.
        "2.jpg": ("train", "apples", (12, 4), [[0.5, 1.0], [11.5, 2.0]],
                  [corners(4, 0, 6, 2), corners(5, 1, 7, 3), corners(6, 2, 8, 4), corners(0, 0, 1, 1)]),
        # Box 0 keeps exactly half its area, box 1 a quarter.
        "3.jpg": ("val", "cars", (10, 6), [[3.0, 3.0], [9.5, 1.0]],
                  [corners(0, 0, 4, 2), [[10, 2], [8, 0], [10, 0], [8, 2]], corners(4, 2, 5, 3)]),
        "4.jpg": ("test", "birds", (6, 8), [[1.0, 1.0], [5.5, 7.5]], THREE),
    })


def task(raw, images, out, name, split="train", letterbox=False):
    annotations, _, classes = prep.load_release(raw)
    for sub in ("images", "letterbox_val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    return {"image_id": prep.image_id(name), "file": name, "path": str(images / name),
            "annotation": annotations[name], "class": classes[name], "split": split,
            "out": str(out), "letterbox": letterbox}


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["prep_fsc147.py", *map(str, argv)])
    prep.main()


def test_crop_recounts_with_half_open_edges_and_writes_native_png(tmp_path):
    raw, images = release(tmp_path)
    result = prep.prepare_one(task(raw, images, tmp_path / "out", "1.jpg"))
    record = result["record"]
    assert record["frame"] == {"kind": "centre_crop", "left": 2, "top": 0, "side": 6, "width": 10, "height": 6}
    # x = 2 (left edge) and x = 7.999 stay; x = 8, x = 1.999 and y = 6 (closing edges) drop.
    assert (record["source_count"], record["count"], record["bin"]) == (6, 3, "le50")
    assert result["density"].dtype == np.float32 and result["density"].shape == (64, 64)
    assert result["density"].sum(dtype=np.float64) == pytest.approx(3, abs=1e-5)
    with Image.open(images / "1.jpg") as source, Image.open(tmp_path / "out/images/fsc147_1.png") as crop:
        assert crop.size == (6, 6)
        assert np.array_equal(np.asarray(crop), np.asarray(source)[:, 2:8])
    assert record["png_sha256"] == prep.file_sha256(tmp_path / "out/images/fsc147_1.png")


def test_box_corners_become_clipped_unit_boxes_with_eligibility(tmp_path):
    assert prep.corners_to_box([[5, 1], [2, 1], [2, 7], [5, 7]]).tolist() == [2, 1, 5, 7]
    raw, images = release(tmp_path)
    record = prep.prepare_one(task(raw, images, tmp_path / "out", "3.jpg", "val"))["record"]
    # Frame left = 2, side = 6: box 0 spans x in [0, 4) -> unit [-1/3, 1/3), keeps half.
    assert record["boxes"][0] == pytest.approx([0.0, 0.0, 2 / 6, 2 / 6])
    assert record["box_kept"] == pytest.approx([0.5, 0.0, 1.0])
    # Box 1 lists its corners out of order and lies past the right edge, x in [8, 10).
    assert record["boxes"][1][0] == pytest.approx(1.0)
    assert record["box_valid"] == [True, False, True]
    assert not record["exemplar_eligible"]
    assert record["exclusions"] == ["exemplar arm: box 1 keeps 0.000 of its area (< 0.5)"]


def test_partial_box_validity_threshold_and_first_three_boxes(tmp_path):
    raw, images = release(tmp_path)
    frame = counting.centre_crop_frame(10, 6)
    boxes, kept, valid, reasons = prep.exemplar_boxes(frame, [corners(0, 0, 4, 2), corners(1, 0, 3, 2)])
    assert boxes[1] == pytest.approx([0.0, 0.0, 1 / 6, 1 / 3])
    assert kept == pytest.approx([0.5, 0.5]) and valid == [True, True] and reasons == []
    boxes, kept, valid, reasons = prep.exemplar_boxes(frame, [corners(0, 0, 3, 2), corners(8, 0, 9, 1)])
    assert kept == pytest.approx([1 / 3, 0.0]) and valid == [False, False] and boxes[0][0] == 0.0
    assert reasons[0] == "box 0 keeps 0.333 of its area (< 0.5)"
    record = prep.prepare_one(task(raw, images, tmp_path / "out", "2.jpg"))["record"]
    assert record["release_boxes"] == 4 and len(record["boxes"]) == 3
    assert record["notes"] == ["release has 4 boxes; boxes 0-2 used"]


def test_zero_count_image_keeps_an_empty_density(tmp_path):
    raw, images = release(tmp_path)
    result = prep.prepare_one(task(raw, images, tmp_path / "out", "2.jpg"))
    assert result["record"]["count"] == 0 and result["record"]["source_count"] == 2
    assert result["density"].sum() == 0 and result["record"]["bin"] == "le50"


def test_letterbox_keeps_every_dot_and_pads_with_the_fill(tmp_path):
    raw, images = release(tmp_path)
    result = prep.prepare_one(task(raw, images, tmp_path / "out", "3.jpg", "val", letterbox=True))
    lb = result["letterbox"]
    assert lb["frame"] == {"kind": "letterbox", "left": 0, "top": -2, "side": 10, "width": 10, "height": 6}
    assert lb["valid"] == pytest.approx([0.0, 0.2, 1.0, 0.8])
    assert (lb["count"], lb["dropped_dots"]) == (2, 0)
    assert result["letterbox_density"].sum(dtype=np.float64) == pytest.approx(2, abs=1e-5)
    assert lb["boxes"][0] == pytest.approx([0.0, 0.2, 0.4, 0.4]) and lb["exemplar_eligible"]
    with Image.open(tmp_path / "out/letterbox_val/fsc147_3.png") as canvas, Image.open(images / "3.jpg") as source:
        pixels = np.asarray(canvas)
        assert canvas.size == (10, 10)
        assert (pixels[:2] == counting.LETTERBOX_FILL).all() and (pixels[8:] == counting.LETTERBOX_FILL).all()
        assert np.array_equal(pixels[2:8], np.asarray(source))


def test_release_check_reports_split_problems():
    def entry(boxes=3):
        return {"points": [[1, 1]], "box_examples_coordinates": [corners(0, 0, 1, 1)] * boxes}

    annotations = {"1.jpg": entry(), "2.jpg": entry(), "3.jpg": entry(2), "1.png": entry(), "5.jpg": entry()}
    classes = {"1.jpg": "a", "2.jpg": "a", "3.jpg": "b", "1.png": "c", "5.jpg": "d"}
    splits = {"train": ["1.jpg"], "val": ["2.jpg", "1.png"], "test": ["3.jpg", "4.jpg"]}
    report = prep.check_release(annotations, splits, classes, {"train": 1, "val": 2, "test": 1}, 3)
    problems = "\n".join(report["problems"])
    assert "classes shared by train and val: ['a']" in problems and not report["classes_disjoint"]
    assert "stem collision 1" in problems and "image_id collision fsc147_1" in problems
    assert "3.jpg has 2 boxes" in problems and "4.jpg (test) has no annotation" in problems
    assert "test has 2 images, the release states 1" in problems
    good = prep.check_release(annotations, {"train": ["1.jpg"], "val": ["5.jpg"], "test": []}, classes,
                              expected={"train": 1, "val": 1, "test": 0}, expected_classes=2)
    assert good["problems"] == [] and good["classes_disjoint"]
    assert good["annotations_outside_splits"] == ["1.png", "2.jpg", "3.jpg"]
    dup = prep.check_release(annotations, {"train": ["1.jpg"], "val": ["1.jpg"], "test": []}, classes, None, None)
    assert "1.jpg appears 2 times across splits" in dup["problems"]


def test_main_writes_aligned_manifest_and_targets_idempotently(tmp_path, monkeypatch):
    raw, images = release(tmp_path)
    out = tmp_path / "out"
    (out / "images").mkdir(parents=True)
    (out / "images" / "fsc147_stale.png").write_bytes(b"old")
    monkeypatch.setattr(prep, "EXPECTED_SPLITS", None)
    monkeypatch.setattr(prep, "EXPECTED_CLASSES", None)
    args = ("--raw", raw, "--images", images, "--out", out, "--workers", 1, "--transport", "test")
    run_main(monkeypatch, *args)

    manifest = json.loads((out / prep.MANIFEST).read_text())
    assert manifest["splits"] == {"train": ["fsc147_1", "fsc147_2"], "val": ["fsc147_3"], "test": ["fsc147_4"]}
    with np.load(out / prep.TARGETS) as targets:
        assert targets.files == list(manifest["images"])
        for iid, record in manifest["images"].items():
            assert targets[iid].sum(dtype=np.float64) == pytest.approx(record["count"], abs=1e-5)
    assert manifest["counts"]["train"]["zero_count_ids"] == ["fsc147_2"]
    assert manifest["counts"]["all"]["eligible"] == 3
    assert manifest["exclusions"][0]["image_id"] == "fsc147_3"
    assert sorted(p.name for p in (out / "images").iterdir()) == [f"fsc147_{i}.png" for i in range(1, 5)]
    lb = json.loads((out / prep.LETTERBOX_MANIFEST).read_text())
    assert lb["ids"] == ["fsc147_3"] and lb["images"]["fsc147_3"]["count"] == 2
    complete = json.loads((out / prep.COMPLETE).read_text())
    assert complete["sha256"][prep.MANIFEST] == prep.file_sha256(out / prep.MANIFEST)

    first = {name: (out / name).read_bytes() for name in complete["sha256"]}
    run_main(monkeypatch, *args)
    assert {name: (out / name).read_bytes() for name in first} == first
    prep.validate(out)
    (out / "images" / "fsc147_1.png").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        prep.validate(out)

    run_main(monkeypatch, *args, "--ids", "1,3")
    fixture = json.loads((out / prep.MANIFEST).read_text())
    assert fixture["fixture"] and list(fixture["images"]) == ["fsc147_1", "fsc147_3"]
    assert sorted(p.stem for p in (out / "images").iterdir()) == ["fsc147_1", "fsc147_3"]

