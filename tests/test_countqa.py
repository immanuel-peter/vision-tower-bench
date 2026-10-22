import hashlib
import json
from dataclasses import dataclass
from math import isqrt
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn.functional import interpolate, softplus

from scripts import countqa_annotate as annotate
from scripts import countqa_eval as cqe
from scripts import prep_countqa as prep
from vtb import counting
from vtb.feature_batch import FeatureBatch
from vtb.probe import Reducer


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
    total = counting.masked_count(torch.from_numpy(density), torch.from_numpy(valid)[None])
    assert total.item() == pytest.approx(2.5)


# ---------------------------------------------------------------- annotation
def tiny_eligibility():
    kept = {
        "countqa_0001_aaaaaaaa": {"question_id": "countqa_0001_aaaaaaaa_q0", "question": "How many limes are there?",
                                  "target": "limes", "answer": 3, "width": 64, "height": 32, "image_sha256": "a"},
        "countqa_0002_bbbbbbbb": {"question_id": "countqa_0002_bbbbbbbb_q1", "question": "How many cups are there?",
                                  "target": "cups", "answer": 60, "width": 32, "height": 64, "image_sha256": "b"},
        "countqa_0003_cccccccc": {"question_id": "countqa_0003_cccccccc_q0", "question": "How many pens are there?",
                                  "target": "pens", "answer": 1, "width": 48, "height": 48, "image_sha256": "c"},
    }
    return {"kept": kept}


def tool_export(eligibility, statuses, sheet_sha="s", eligibility_sha="e", annotator="Ann"):
    items = []
    for (image_id, entry), status in zip(sorted(eligibility["kept"].items()), statuses):
        items.append({"question_id": entry["question_id"], "image_id": image_id, "status": status,
                      "box_px": [2.0, 3.0, 10.0, 12.0] if status == "boxed" else None,
                      "annotated_at": "t", "note": ""})
    return {"format": annotate.EXPORT_FORMAT, "sheet_sha256": sheet_sha, "eligibility_sha256": eligibility_sha,
            "annotator": annotator, "exported_at": "t", "items": items}


def test_sheet_and_pending_exemplars_start_empty():
    eligibility = tiny_eligibility()
    rows = annotate.sheet_rows(eligibility)
    assert [r["status"] for r in rows] == ["pending"] * 3 and rows[0]["box_x0_px"] == ""
    assert annotate.sheet_csv(rows).splitlines()[0].split(",") == list(annotate.SHEET_COLUMNS)
    pending = annotate.pending_exemplars(eligibility, "e", "s")
    assert pending["status"] == "pending"
    assert all(item["box_px"] is None and item["status"] == "pending" for item in pending["items"].values())


def test_finalize_requires_every_question_resolved_and_maps_boxes():
    eligibility = tiny_eligibility()
    with pytest.raises(ValueError, match="still 'pending'"):
        annotate.finalize(tool_export(eligibility, ["boxed", "pending", "boxed"]), "x", eligibility, "e", "s", "Ann")
    with pytest.raises(ValueError, match="different sheet"):
        annotate.finalize(tool_export(eligibility, ["boxed"] * 3, sheet_sha="other"), "x", eligibility, "e", "s", "Ann")
    with pytest.raises(ValueError, match="annotator"):
        annotate.finalize(tool_export(eligibility, ["boxed"] * 3), "x", eligibility, "e", "s", " ")
    done = annotate.finalize(tool_export(eligibility, ["boxed", "no_clean_instance", "boxed"]), "x",
                             eligibility, "e", "s", "Ann", now="t")
    assert done["status"] == "approved" and done["counts"]["no_clean_instance"] == 1
    item = done["items"]["countqa_0001_aaaaaaaa_q0"]
    assert item["box_source"] == "human"
    assert item["box_source_unit"] == pytest.approx([2 / 64, 3 / 32, 10 / 64, 12 / 32])
    # 64 x 32 letterboxed on a 64 square: 16 px of padding on top
    assert item["box_letterbox_unit"] == pytest.approx([2 / 64, 19 / 64, 10 / 64, 28 / 64])
    with pytest.raises(ValueError, match="leaves"):
        annotate.validate_box([0, 0, 65, 10], 64, 32)


def test_html_tool_embeds_the_items_and_hashes():
    eligibility = tiny_eligibility()
    names = {image_id: f"{image_id}.jpg" for image_id in eligibility["kept"]}
    html = annotate.render_html(annotate.tool_items(eligibility, names), "e", "s")
    assert "__DATA__" not in html and '"sheet_sha256": "s"' in html
    assert "images/countqa_0001_aaaaaaaa.jpg" in html and annotate.EXPORT_FORMAT in html


# ---------------------------------------------------------------- eval fixtures
@dataclass
class StandInReadout:
    """Matches the contract's Readout interface."""

    head: nn.Module
    reducer: Reducer | None
    conditioned: bool
    meta: dict

    def density(self, tokens, boxes=None, valid_rects=None):
        x = tokens.float()
        if self.reducer is not None:
            x = self.reducer(x)
        n, t, d = x.shape
        side = isqrt(t)
        fmap = x.transpose(1, 2).reshape(n, d, side, side)
        if self.conditioned:
            if boxes is None:
                raise ValueError("conditioned readout needs boxes")
            valid = None
            if valid_rects is not None:
                valid = torch.from_numpy(np.stack([counting.token_valid(r, side) for r in valid_rects]))
            fmap = counting.condition(fmap, boxes, valid)
        out = softplus(self.head(fmap))
        grid = counting.TARGET_GRID
        return interpolate(out, size=(grid, grid), mode="nearest") * (side / grid) ** 2


def stand_in(conditioned, matched, dim=6, width=4, bias=0.0, seed=0):
    torch.manual_seed(seed)
    reducer = Reducer(basis=torch.randn(dim, width), mean=torch.randn(1, dim)) if matched else None
    head = nn.Conv2d((width if matched else dim) + int(conditioned), 1, 1)
    with torch.no_grad():
        head.weight.mul_(0.1)
        head.bias.fill_(bias)
    return StandInReadout(head.eval(), reducer, conditioned, {"protocol": counting.PROTOCOL, "seed": seed})


def build_fixture(tmp_path, approved=True, side=4, dim=6):
    eligibility = tiny_eligibility()
    manifest = {"eligibility_sha256": "e", "source": {"repo": prep.REPO, "revision": prep.REVISION,
                                                      "license": prep.LICENSE, "citation": prep.CITATION},
                "transfer_preprocessing": prep.TRANSFER_PREPROCESSING, "images": {}}
    for image_id, entry in eligibility["kept"].items():
        frame = counting.letterbox_frame(entry["width"], entry["height"])
        manifest["images"][image_id] = {"width": entry["width"], "height": entry["height"],
                                        "frame": frame.as_dict(), "question_id": entry["question_id"],
                                        "answer": entry["answer"]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    if approved:
        exemplars = annotate.finalize(tool_export(eligibility, ["boxed", "boxed", "no_clean_instance"]), "x",
                                      eligibility, "e", "s", "Ann", now="t")
    else:
        exemplars = annotate.pending_exemplars(eligibility, "e", "s")
    exemplar_path = tmp_path / "exemplars.json"
    exemplar_path.write_text(json.dumps(exemplars))
    features = tmp_path / "features"
    features.mkdir()
    ids = sorted(eligibility["kept"])[::-1]  # cache order differs from sorted order on purpose
    torch.manual_seed(1)
    for stage, grid in (("tower", side), ("merged", side // 2)):
        FeatureBatch(tokens=torch.randn(len(ids), grid * grid, dim).to(torch.bfloat16), image_ids=ids,
                     model_id="toy", stage=stage, layer_index=2, num_layers=2, resolution=32).save(
            features / f"{stage}_L02_00000.safetensors")
    (features / "complete.json").write_text(json.dumps({
        "model_id": "toy", "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }))
    readouts = tmp_path / "toy"
    for arm in ("exemplar", "free"):
        for stage in ("tower", "merged"):
            (readouts / f"{arm}_matched" / f"{stage}_L02").mkdir(parents=True)
    return SimpleNamespace(
        features=features, readouts=readouts, arm="exemplar", variant="matched", stages=None, seeds=[0, 1],
        manifest=manifest_path, exemplars=exemplar_path, device="cpu", batch_size=2, resamples=200,
        confidence=0.95, bootstrap_seed=0, allow_incomplete_cache=False,
    )


class Loader:
    def __init__(self, conditioned, bias=0.0):
        self.conditioned, self.bias, self.loaded = conditioned, bias, []

    def __call__(self, directory, seed, device):
        readout = stand_in(self.conditioned, True, bias=self.bias, seed=seed)
        self.loaded.append(readout)
        return readout


def readout_tensors(readout):
    tensors = list(readout.head.state_dict().values()) + [readout.reducer.mean, readout.reducer.basis]
    return [t.clone() for t in tensors]


# ---------------------------------------------------------------- eval tests
def test_conditioned_eval_refuses_unapproved_annotations(tmp_path):
    args = build_fixture(tmp_path, approved=False)
    loader = Loader(conditioned=True)
    with pytest.raises(cqe.Refused, match="not 'approved'"):
        cqe.evaluate(args, loader)
    assert loader.loaded == []  # refused before any readout was loaded
    with pytest.raises(SystemExit, match="refused"):
        cqe.main(["--features", str(args.features), "--readouts", str(args.readouts), "--arm", "exemplar",
                  "--manifest", str(args.manifest), "--exemplars", str(args.exemplars)])


def test_suggested_boxes_are_never_used(tmp_path):
    args = build_fixture(tmp_path)
    data = json.loads(args.exemplars.read_text())
    item = data["items"]["countqa_0001_aaaaaaaa_q0"]
    item["box_source"], item["suggested_unreviewed"] = "ai_suggested", item["box_px"]
    args.exemplars.write_text(json.dumps(data))
    with pytest.raises(cqe.Refused, match="source"):
        cqe.evaluate(args, Loader(conditioned=True))


def test_eval_keeps_readouts_frozen_and_writes_contract_rows(tmp_path, monkeypatch):
    def no_fitting(*_args, **_kwargs):
        raise AssertionError("nothing may be fit on CountQA")

    monkeypatch.setattr(Reducer, "fit", classmethod(no_fitting))
    args = build_fixture(tmp_path)
    loader = Loader(conditioned=True)
    snapshots = []

    def tracking(directory, seed, device):
        readout = loader(directory, seed, device)
        snapshots.append((readout, readout_tensors(readout), readout.head.training))
        return readout

    payload = cqe.evaluate(args, tracking)
    assert len(snapshots) == 4  # two Stages x two seeds
    for readout, before, training in snapshots:
        after = readout_tensors(readout)
        assert all(torch.equal(a, b) for a, b in zip(before, after))
        assert readout.head.training is training is False
    assert payload["label"] == cqe.LABEL
    assert payload["primary_cell"] == "tower_L02"
    assert payload["dataset"]["excluded"] == {"countqa_0003_cccccccc": "no_clean_instance"}
    assert [c["key"] for c in payload["cells"]] == ["tower_L02", "merged_L02"]
    assert payload["cells"][0]["metrics"]["images"] == 2
    assert "tower_L02_vs_merged_L02" in payload["stage_differences"]
    out = tmp_path / "toy_count_exemplar_matched.json"
    cqe.write(payload, out)
    lines = out.with_suffix(".predictions.tsv").read_text().splitlines()
    assert lines[0].split("\t") == list(cqe.TSV_COLUMNS)
    assert len(lines) == 1 + 2 * 2 * 2  # Stages x seeds x images
    first = lines[1].split("\t")
    assert first[1] == "countqa" and first[4] == "exemplar"
    assert json.loads(out.read_text())["label"] == cqe.LABEL


def test_eval_arm_must_match_the_readout(tmp_path):
    args = build_fixture(tmp_path)
    with pytest.raises(cqe.Refused, match="conditioned=False"):
        cqe.evaluate(args, Loader(conditioned=False))


def test_free_arm_counts_only_the_valid_area(tmp_path):
    args = build_fixture(tmp_path)
    args.arm = "free"
    readout = stand_in(False, True)
    with torch.no_grad():
        readout.head.weight.zero_()
        readout.head.bias.fill_(2.0)  # constant density everywhere, padding included
    payload = cqe.evaluate(args, lambda d, s, dev: readout)
    per_cell = softplus(torch.tensor(2.0)).item() * (4 / 64) ** 2
    cell = payload["_cells"][0]
    expected = {"countqa_0001_aaaaaaaa": 64 * 64 * 0.5, "countqa_0002_bbbbbbbb": 64 * 64 * 0.5,
                "countqa_0003_cccccccc": 64 * 64}
    assert cell["predicted"][0] == pytest.approx([per_cell * expected[i] for i in cell["image_ids"]], rel=1e-5)
    assert payload["dataset"]["evaluated_images"] == 3
    assert payload["cells"][0]["metrics_common_exemplar_subset"]["images"] == 2


def test_eval_needs_a_complete_cache(tmp_path):
    args = build_fixture(tmp_path)
    (args.features / "complete.json").unlink()
    with pytest.raises(cqe.Refused, match="complete.json"):
        cqe.evaluate(args, Loader(conditioned=True))


# ---------------------------------------------------------------- metrics and bootstrap
def test_metrics_round_half_up_and_bin_on_the_true_answer():
    true = np.array([3.0, 60.0, 250.0])
    predicted = np.array([[2.5, 60.4, 249.49], [3.49, 59.5, 251.0]])
    out = cqe.summarise(predicted, true)
    assert cqe.exact(np.array([2.5, 2.49, 3.5]), np.array([3, 2, 3])).tolist() == [1.0, 1.0, 0.0]
    assert out["exact_match_by_seed"] == pytest.approx([2 / 3, 2 / 3])
    errors = np.abs(predicted - true)
    assert out["mae"] == pytest.approx(errors.mean())
    assert out["rmse"] == pytest.approx(np.sqrt((errors ** 2).mean()))
    assert out["by_bin"]["le50"]["images"] == 1 and out["by_bin"]["50to200"]["images"] == 1
    assert out["by_bin"]["gt200"]["exact_match"] == pytest.approx(0.0)
    assert out["by_bin"]["le50"]["mae"] == pytest.approx((0.5 + 0.49) / 2)


def test_stage_bootstrap_pairs_identical_ids_and_seeds():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    base = {"image_ids": ["a", "b", "c", "d"], "seeds": [0, 1]}
    first = base | {"key": "tower_L2", "predicted": np.tile(true, (2, 1))}
    second = base | {"key": "merged_L2", "predicted": np.tile(true + 1.0, (2, 1))}
    out = cqe.stage_differences([first, second], true, resamples=100, confidence=0.95, seed=0)
    diff = out["tower_L2_vs_merged_L2"]
    assert diff["abs_error"]["point_estimate"] == pytest.approx(1.0)  # tower has 1.0 lower error
    assert diff["abs_error"]["difference"] == "second_minus_first"
    assert diff["exact_match"]["point_estimate"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="share"):
        cqe.stage_differences([first, second | {"seeds": [0, 2]}], true, 100, 0.95, 0)
