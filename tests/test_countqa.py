import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from scripts import prep_countqa as prep
from vtb import counting


def make_row(index, questions, answers, focused=True, image=b"img", width=40, height=20):
    data = image + bytes([index % 256])
    return prep.Row(
        index=index, sha256=hashlib.sha256(data).hexdigest(), image_bytes=data, image_format="JPEG",
        mode="RGB", width=width, height=height, questions=questions, answers=answers, objects=["x"],
        categories=["c"], is_focused=focused, shard="data/test-00000-of-00013.parquet",
    )


# ---------------------------------------------------------------- filter rules
@pytest.mark.parametrize(
    "question, reasons, target",
    [
        ("How many limes are there?", [], "limes"),
        ("How many books are visible?", [], "books"),
        ("What is the total number of peanuts?", [], "peanuts"),
        ("How many cups are there in total?", [], "cups"),
        ("How many total bottles are there?", [], "bottles"),
        ("How many objects are bowls?", [], "bowls"),
        ("How many pieces of wood are there?", ["attribute_qualifier"], "pieces of wood"),
        ("How many yellow flowers are there", ["colour_qualifier"], "yellow flowers"),
        ("How many cups are on the left?", ["location_qualifier", "part_of_qualifier", "unparsed_question_form"], None),
        ("How many books are in the foreground?", ["location_qualifier", "part_of_qualifier", "unparsed_question_form"], None),
        ("How many tiles are on the wall with the shower?", ["part_of_qualifier", "unparsed_question_form"], None),
        ("How many magnets are on the fridge?", ["part_of_qualifier", "unparsed_question_form"], None),
        ("How many red and black thumb pins are there?", ["combination", "colour_qualifier"], "red and black thumb pins"),
        ("How many packets/boxes are there?", ["combination"], "packets/boxes"),
        ("How many more action cards are there than cat cards?", ["comparison", "unparsed_question_form"], None),
        ("How many balls are not green?", ["negation", "colour_qualifier", "unparsed_question_form"], None),
        ("Ignore the tubs in the back. How many tubs are there in the foreground?",
         ["negation", "location_qualifier", "part_of_qualifier", "unparsed_question_form"], None),
        ("How many small screws are there?", ["attribute_qualifier"], "small screws"),
        ("How many horizontal metal rods are there?", ["attribute_qualifier"], "horizontal metal rods"),
        ("How many objects are there?", ["generic_target"], "objects"),
    ],
)
def test_question_rules_flag_qualifiers_and_parse_the_target(question, reasons, target):
    found, parsed = prep.question_rules(question)
    assert found == reasons
    assert parsed == target


def test_answer_rules_require_a_positive_integer():
    assert prep.answer_rules(" 12 ") == ([], 12)
    assert prep.answer_rules("0") == (["zero_answer"], 0)
    assert prep.answer_rules("about 5")[0] == ["non_integer_answer"]


def test_decide_keeps_one_question_per_focused_image_and_records_reasons():
    rows = [
        make_row(0, ["How many cups are on the left?", "How many cups are there in total?",
                     "What is the total number of cups?"], ["3", "12", "12"]),
        make_row(1, ["How many limes are there?"], ["19"], focused=False),
        make_row(2, ["How many red pins are there?", "How many hearts are there?"], ["4", "3"]),
    ]
    review = {f"{rows[2].image_id}_q1": ("agent_subtype_selection", "card suit")}
    records = {r["question_id"]: r for r in prep.decide(rows, review)}
    first = rows[0].image_id
    assert records[f"{first}_q0"]["status"] == "dropped"
    assert records[f"{first}_q1"]["status"] == "kept" and records[f"{first}_q1"]["target"] == "cups"
    assert records[f"{first}_q2"]["reasons"] == ["not_selected_one_question_per_image"]
    assert records[f"{rows[1].image_id}_q0"]["reasons"] == ["not_focused"]
    assert records[f"{rows[2].image_id}_q1"]["stage"] == "agent_review"
    images = prep.image_status(list(records.values()))
    assert images[first]["status"] == "kept"
    assert images[rows[1].image_id]["reason"] == "not_focused"
    assert images[rows[2].image_id]["reason"] == "no_eligible_question"


def test_review_entries_must_match_a_rule_passing_question():
    rows = [make_row(0, ["How many red pins are there?"], ["4"])]
    with pytest.raises(ValueError, match="agent review"):
        prep.decide(rows, {f"{rows[0].image_id}_q0": ("agent_mixed_category", "")})


def test_duplicate_images_with_conflicting_answers_are_dropped():
    a = make_row(3, ["How many pipes are there?"], ["10"])
    b = make_row(36, ["How many pipes are there?"], ["14"])
    b.sha256 = a.sha256
    c = make_row(40, ["How many pipes are there?"], ["10"])
    d = make_row(41, ["How many pipes are there?"], ["10"])
    d.sha256 = c.sha256
    assert prep.duplicate_reasons([a, b, c, d]) == {
        3: "duplicate_image_conflicting_answers", 36: "duplicate_image_conflicting_answers",
        41: "duplicate_image",
    }


# ---------------------------------------------------------------- stable IDs
def test_stable_ids_combine_row_index_and_content_hash():
    sha = hashlib.sha256(b"x").hexdigest()
    assert prep.stable_id(7, sha) == f"countqa_0007_{sha[:8]}"
    assert prep.question_id(prep.stable_id(7, sha), 2).endswith("_q2")
    rows = [make_row(0, ["How many limes are there?"], ["1"]), make_row(1, ["How many limes are there?"], ["1"])]
    prep.check_ids(rows)
    rows[1].sha256 = rows[0].sha256[:8] + "f" * 56
    with pytest.raises(ValueError, match="prefix"):
        prep.check_ids(rows)


def test_frozen_eligibility_refuses_to_change(tmp_path):
    rows = [make_row(0, ["How many limes are there?"], ["19"])]
    source = {"repo": prep.REPO, "revision": prep.REVISION}
    prep.write_frozen(tmp_path, prep.decide(rows, {}), rows, source)
    lines = (tmp_path / "eligibility.sha256").read_text().splitlines()
    assert [line.split()[1] for line in lines] == ["eligibility.tsv", "eligibility.json"]
    assert prep.write_frozen(tmp_path, prep.decide(rows, {}), rows, source)["counts"]["kept_images"] == 1
    rows[0].answers = ["20"]
    with pytest.raises(SystemExit, match="frozen"):
        prep.write_frozen(tmp_path, prep.decide(rows, {}), rows, source)


# ---------------------------------------------------------------- letterbox
def test_letterbox_image_centres_the_whole_image_on_the_fill():
    image = Image.new("RGB", (6, 3), (255, 0, 0))
    canvas, frame, alpha = prep.letterbox_image(image)
    assert canvas.size == (6, 6) and alpha == "none"
    assert (frame.left, frame.top, frame.side) == (0, -1, 6)
    pixels = np.asarray(canvas)
    assert (pixels[0] == counting.LETTERBOX_FILL).all() and (pixels[4:] == counting.LETTERBOX_FILL).all()
    assert (pixels[1:4] == (255, 0, 0)).all()


def test_transparent_pixels_go_over_the_fill_and_opaque_alpha_is_dropped():
    opaque = Image.new("RGBA", (2, 2), (10, 20, 30, 255))
    assert prep.to_rgb(opaque)[1] == "alpha_opaque_dropped"
    clear = Image.new("RGBA", (2, 2), (10, 20, 30, 0))
    rgb, how = prep.to_rgb(clear)
    assert how == "alpha_composited_over_fill"
    assert rgb.getpixel((0, 0)) == counting.LETTERBOX_FILL


def test_letterbox_frame_maps_source_boxes_and_masks_the_padding():
    frame = counting.letterbox_frame(width=400, height=200)
    box = frame.boxes(np.array([[0.0, 0.0, 100.0, 50.0]]))[0]
    assert box.tolist() == pytest.approx([0.0, 100 / 400, 0.25, 150 / 400])
    clipped, kept = counting.clip_box(box)
    assert kept == pytest.approx(1.0)
    # all density in the top padding band is excluded from the count
    density = np.zeros((1, 1, 64, 64), dtype=np.float32)
    density[0, 0, :16, :] = 1.0
    density[0, 0, 32, 10] = 2.5
    valid = counting.coverage(frame.valid, 64)
    import torch
    total = counting.masked_count(torch.from_numpy(density), torch.from_numpy(valid)[None])
    assert total.item() == pytest.approx(2.5)
