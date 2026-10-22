#!/usr/bin/env python3
"""Prepare the CountQA transfer subset for the frozen FSC-147 count readouts (issue #10).

Three steps, each reproducible from the pinned dataset revision:

1. ``download``: fetch Jayant-Sravan/CountQA at REVISION into $VTB_WORK/data/countqa/raw/
   and check every parquet file against the sha256 the Hub reported for that revision.
2. ``eligibility``: keep one counting question per focused image. Deterministic text rules
   run first, then a recorded agent review of the question text (never a human review).
   The result is frozen to results/counting/countqa/ with a sha256 before any model output
   exists, and a rerun that would change it fails.
3. ``letterbox``: write each kept image as a full-frame letterbox square PNG at native
   resolution plus a manifest of frames, answers and hashes.

CountQA answers count the whole image, so the transfer frame is
``counting.letterbox_frame``: the image is centred on a max-side square and the padding is
``counting.LETTERBOX_FILL``. FSC-147 training uses the centre crop. Counts are summed over
the valid (non-padding) area only.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from vtb import counting

REPO = "Jayant-Sravan/CountQA"
REVISION = "f92cc6fe46542c61e2916e3d2ae9a911e2216b1a"
LICENSE = "CC-BY-4.0"
CITATION = (
    "Tamarapalli, Grover, Pande, Yerramilli. CountQA: How Well Do MLLMs Count in the Wild? "
    "arXiv:2508.06585 (2025). Dataset Jayant-Sravan/CountQA, CC-BY-4.0."
)
# sha256 of each file at REVISION, as reported by the Hub (LFS oid for the parquet shards).
PARQUET_SHA256 = {
    "data/test-00000-of-00013.parquet": "c51b7c6f36cffa0ecb2b378565eb0ce391ddc2655fa802fbe4cbe735ac1e4b5a",
    "data/test-00001-of-00013.parquet": "206157403ebadd250d627c357d4d268a7851705197172ead4cf3fbe43f5d5bfd",
    "data/test-00002-of-00013.parquet": "ffcedd3a6b96d03996d3c1b43cc2cdeed05cef1c50eec8e4373a66d5c25a9943",
    "data/test-00003-of-00013.parquet": "5d8bccf5e47848dfc2ccfcab28db863fa5b1fdf1f9613094b1ef0c8b1f695357",
    "data/test-00004-of-00013.parquet": "f06f7373b2dd5e42ecfa88dde65dcffacbd08e30c8d3183df5afc06b9ba9741c",
    "data/test-00005-of-00013.parquet": "4453bb5e2ff0dcb5119063ac5b37cbca56d19b246a1c0e9824e4d3611dd564b5",
    "data/test-00006-of-00013.parquet": "5deda1d3efcd2ce832b759c9399896ce0952d2d0f3a6b648fcb2afbcda0bab0d",
    "data/test-00007-of-00013.parquet": "11aaaa22c129f72df713893af7f0ef5352b818f361adb07b0975cc24c2e7815a",
    "data/test-00008-of-00013.parquet": "fe064ea12283b37ae2f7da59ad303404af9167a8da30538a41d13064cfd00ff1",
    "data/test-00009-of-00013.parquet": "f216b82216bd4ee05f4b868feaef84e98f42dac1ee7676554e091217e3913c99",
    "data/test-00010-of-00013.parquet": "aff31256389095b78ffdb7ab716ad03e3899977b44786242c2ea54d5660b9581",
    "data/test-00011-of-00013.parquet": "69423346de3eabc7089fc8e57e3fa857c8fa8df4da83fc6187e8eea4caa26519",
    "data/test-00012-of-00013.parquet": "c66241ffe4fdb3786dfa13821c76f526543348a446ab1101f895712866143e3a",
}
RULES_VERSION = "countqa-eligibility-v1"
ID_PREFIX = "countqa_"
SHA_PREFIX = 8
RESULTS_DIR = Path("results/counting/countqa")
TRANSFER_PREPROCESSING = (
    "CountQA answers refer to the whole image, so each kept image is placed whole on a "
    "max(w, h) square canvas (counting.letterbox_frame, integer centring offsets) filled with "
    "counting.LETTERBOX_FILL and saved as PNG at native resolution; the adapter then resizes "
    "that square to the benchmark resolution. FSC-147 heads were trained on the central "
    "min-side square (counting.centre_crop_frame), so CountQA adds padding the heads never "
    "saw and a different object scale; predicted density is summed only over the valid "
    "(non-padding) area, counting.masked_count(density, counting.coverage(frame.valid, 64)). "
    "Exemplar boxes are mapped with frame.boxes, never centre-cropped."
)

# ---------------------------------------------------------------- deterministic rules
COLOURS = {
    "white", "black", "red", "blue", "green", "yellow", "gold", "golden", "silver", "gray",
    "grey", "pink", "purple", "orange", "brown", "clear", "multicolored", "multicoloured",
    "colored", "coloured", "beige", "violet", "transparent", "dark", "light",
}
LOCATIONS = {
    "left", "right", "center", "centre", "middle", "top", "bottom", "front", "back",
    "foreground", "background", "upper", "lower", "above", "below", "under", "underneath",
    "behind", "beside", "near", "corner", "side", "inside", "outside", "between",
}
ATTRIBUTES = {
    # material
    "metal", "metallic", "plastic", "wooden", "wood", "glass", "steel", "ceramic",
    # size
    "small", "large", "big", "tiny", "little", "thick", "thin", "long", "short", "tall",
    # orientation
    "horizontal", "vertical", "diagonal", "upright",
    # shape adjectives
    "square", "round", "circular", "triangular", "rectangular", "flat",
}
GENERIC_TARGETS = {"objects", "object", "items", "item", "things", "thing", "stuff"}
COMBINATION = re.compile(r"\b(and|or|either|combined|combine|plus|add|together|sum)\b|[/&+]")
COMPARISON = re.compile(r"\b(more|fewer|less|than|difference|minus|subtract)\b")
NEGATION = re.compile(r"\b(not|except|excluding|exclude|ignore|ignoring|without|besides)\b|n't|other than")
PART_OF = re.compile(
    r"\b(on|in|inside|of|at|with|within|from|attached to|belonging to)\s+"
    r"(the|a|an|this|that|these|those|its|their)\b"
)
TARGET_PATTERNS = (
    re.compile(r"^how many (?:total )?(?P<target>.+?) (?:are|is) (?:there|visible|present|shown)(?: in total)?$"),
    re.compile(r"^what is the (?:total )?number of (?P<target>.+?)$"),
    re.compile(r"^how many (?:objects|items|things) are (?P<target>.+?)$"),
)

# Reason codes. Deterministic rules come first, the agent review second, selection last.
REASONS = {
    "not_focused": "rule: is_focused is false",
    "duplicate_image_conflicting_answers": "rule: identical image bytes appear in several rows "
    "whose answers to the same question disagree",
    "duplicate_image": "rule: identical image bytes in an earlier kept row",
    "non_integer_answer": "rule: answer is not a non-negative integer",
    "zero_answer": "rule: answer is 0, so no instance exists to box",
    "combination": "rule: sums or unions across object types or subsets (and/or/combined/...)",
    "comparison": "rule: comparison or arithmetic between counts",
    "negation": "rule: negated or excluding selection",
    "colour_qualifier": "rule: colour qualifier",
    "location_qualifier": "rule: location, foreground/background or side qualifier",
    "part_of_qualifier": "rule: part-of or container phrase (on/in/of/with the ...)",
    "attribute_qualifier": "rule: material, size, orientation or shape adjective",
    "generic_target": "rule: target is a generic word such as objects",
    "unparsed_question_form": "rule: not of the form 'How many X are there/visible' or "
    "'What is the (total) number of X'",
    "agent_subtype_selection": "agent review: asks for a subtype (brand, card suit, card kind, "
    "towel kind) of objects that the question set shows co-occur in the image",
    "agent_mixed_category": "agent review: target is a category spanning several object types",
    "agent_part_of_qualifier": "agent review: compound noun restating a part-of qualifier",
    "not_selected_one_question_per_image": "selection: another eligible question on this image "
    "comes first in release order",
    "no_eligible_question": "image: no question on this image survives the rules and review",
}

# Agent review of the question text only (no images were viewed). Recorded per question ID
# and applied after the deterministic rules. This is not a human review.
AGENT_REVIEW = {
    # hand towels vs bath towels (the image also asks for both and the total)
    "countqa_0107_c22d81be_q0": ("agent_subtype_selection", "hand towels; the image also has bath towels"),
    "countqa_0107_c22d81be_q1": ("agent_subtype_selection", "bath towels; the image also has hand towels"),
    # card kinds
    "countqa_0090_5c693125_q1": ("agent_subtype_selection", "cat cards among action cards"),
    "countqa_0090_5c693125_q2": ("agent_subtype_selection", "action cards among cat cards"),
    "countqa_0145_6745f72b_q1": ("agent_subtype_selection", "cat cards among action cards"),
    "countqa_0145_6745f72b_q2": ("agent_subtype_selection", "action cards among cat cards"),
    # playing-card suits and jokers
    "countqa_0271_9c446557_q0": ("agent_subtype_selection", "card suit (hearts)"),
    "countqa_0271_9c446557_q2": ("agent_subtype_selection", "card suit (spades)"),
    "countqa_0271_9c446557_q3": ("agent_subtype_selection", "card suit (clubs)"),
    "countqa_0455_bb365c4e_q1": ("agent_subtype_selection", "jokers among other cards"),
    "countqa_0455_bb365c4e_q2": ("agent_subtype_selection", "card suit (hearts)"),
    "countqa_0455_bb365c4e_q3": ("agent_subtype_selection", "card suit (diamonds)"),
    "countqa_0455_bb365c4e_q4": ("agent_subtype_selection", "card suit (spades)"),
    "countqa_0455_bb365c4e_q5": ("agent_subtype_selection", "card suit (clubs)"),
    # brands among other brands of the same product
    "countqa_0510_a58151ee_q1": ("agent_subtype_selection", "Crest among other toothpaste brands"),
    "countqa_0510_a58151ee_q2": ("agent_subtype_selection", "Colgate among other toothpaste brands"),
    "countqa_0510_a58151ee_q3": ("agent_subtype_selection", "Sensodyne among other toothpaste brands"),
    "countqa_0669_9ee1194f_q1": ("agent_subtype_selection", "Crest among other toothpaste brands"),
    "countqa_0669_9ee1194f_q2": ("agent_subtype_selection", "Colgate among other toothpaste brands"),
    "countqa_0669_9ee1194f_q3": ("agent_subtype_selection", "Sensodyne among other toothpaste brands"),
    "countqa_0530_224845e5_q1": ("agent_subtype_selection", "Dorito tubes among Funyun tubes"),
    "countqa_0530_224845e5_q2": ("agent_subtype_selection", "Funyun tubes among Dorito tubes"),
    # categories spanning object types
    "countqa_0396_12449f70_q0": ("agent_mixed_category", "fruits can mix several fruit types"),
    "countqa_0891_844b3c76_q0": ("agent_mixed_category", "fruits can mix several fruit types"),
    "countqa_0226_9a999202_q0": ("agent_mixed_category", "cutlery mixes forks, knives and spoons"),
    # "number of shower holes" restates the rule-dropped "holes ... on the shower"
    "countqa_0978_88b20395_q1": ("agent_part_of_qualifier", "shower holes restates holes on the shower"),
}


@dataclass
class Row:
    """One dataset row (one image) at the pinned revision."""

    index: int
    sha256: str
    image_bytes: bytes = field(repr=False)
    image_format: str
    mode: str
    width: int
    height: int
    questions: list[str]
    answers: list[str]
    objects: list[str]
    categories: list[str]
    is_focused: bool
    shard: str

    @property
    def image_id(self) -> str:
        return stable_id(self.index, self.sha256)


def stable_id(index: int, sha256: str) -> str:
    """``countqa_<row index, 4 digits>_<first 8 hex of the image sha256>``.

    The row index is the position in the concatenated parquet shards (sorted by name) at
    REVISION. The hash prefix ties the ID to the image content, so a reordered or changed
    release cannot silently reuse an ID.
    """
    return f"{ID_PREFIX}{index:04d}_{sha256[:SHA_PREFIX]}"


def question_id(image_id: str, k: int) -> str:
    return f"{image_id}_q{k}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise(question: str) -> str:
    text = question.strip().lower().replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.rstrip("?.! ").strip()


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text))


def parse_target(question: str) -> str | None:
    text = normalise(question)
    for pattern in TARGET_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group("target").strip()
    return None


def question_rules(question: str) -> tuple[list[str], str | None]:
    """Deterministic reason codes for one question and its parsed target (None if unparsed)."""
    text = normalise(question)
    tokens = words(text)
    reasons = []
    if COMBINATION.search(text):
        reasons.append("combination")
    if COMPARISON.search(text):
        reasons.append("comparison")
    if NEGATION.search(text):
        reasons.append("negation")
    if tokens & COLOURS:
        reasons.append("colour_qualifier")
    if tokens & LOCATIONS or "next to" in text:
        reasons.append("location_qualifier")
    if PART_OF.search(text):
        reasons.append("part_of_qualifier")
    if tokens & ATTRIBUTES:
        reasons.append("attribute_qualifier")
    target = parse_target(question)
    if target is None:
        reasons.append("unparsed_question_form")
    elif target in GENERIC_TARGETS:
        reasons.append("generic_target")
    return reasons, target


def answer_rules(answer: str) -> tuple[list[str], int | None]:
    text = answer.strip()
    if not re.fullmatch(r"\d+", text):
        return ["non_integer_answer"], None
    value = int(text)
    return (["zero_answer"] if value == 0 else []), value


def duplicate_reasons(rows: list[Row]) -> dict[int, str]:
    """Row index -> duplicate reason for rows sharing identical image bytes."""
    groups: dict[str, list[Row]] = {}
    for row in rows:
        groups.setdefault(row.sha256, []).append(row)
    out = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        answers: dict[str, set[str]] = {}
        for row in group:
            for q, a in zip(row.questions, row.answers):
                answers.setdefault(normalise(q), set()).add(a.strip())
        if any(len(values) > 1 for values in answers.values()):
            for row in group:
                out[row.index] = "duplicate_image_conflicting_answers"
        else:
            for row in sorted(group, key=lambda r: r.index)[1:]:
                out[row.index] = "duplicate_image"
    return out


def check_ids(rows: list[Row]) -> None:
    ids = [row.image_id for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("stable image IDs collide")
    prefixes: dict[str, str] = {}
    for row in rows:
        prefix = row.sha256[:SHA_PREFIX]
        if prefixes.setdefault(prefix, row.sha256) != row.sha256:
            raise ValueError(f"sha256 prefix {prefix} is shared by different images")


def decide(rows: list[Row], review: dict[str, tuple[str, str]] | None = None) -> list[dict]:
    """One record per question with status kept/dropped, the stage that decided and reasons."""
    review = AGENT_REVIEW if review is None else review
    check_ids(rows)
    duplicates = duplicate_reasons(rows)
    records = []
    for row in rows:
        if len(row.questions) != len(row.answers):
            raise ValueError(f"row {row.index}: {len(row.questions)} questions, {len(row.answers)} answers")
        image_records = []
        for k, (question, answer) in enumerate(zip(row.questions, row.answers)):
            qid = question_id(row.image_id, k)
            rule_reasons, target = question_rules(question)
            answer_reasons, value = answer_rules(answer)
            reasons = []
            if not row.is_focused:
                reasons.append("not_focused")
            if row.index in duplicates:
                reasons.append(duplicates[row.index])
            reasons += answer_reasons + rule_reasons
            stage = "rule" if reasons else None
            note = ""
            if not reasons and qid in review:
                code, note = review[qid]
                reasons, stage = [code], "agent_review"
            image_records.append({
                "question_id": qid, "image_id": row.image_id, "row": row.index, "question_index": k,
                "image_sha256": row.sha256, "width": row.width, "height": row.height,
                "is_focused": row.is_focused, "question": question.strip(), "target": target or "",
                "answer": answer.strip(), "answer_int": value, "objects": [o.strip() for o in row.objects],
                "status": "dropped" if reasons else "eligible", "stage": stage or "", "reasons": reasons,
                "review_note": note,
            })
        eligible = [r for r in image_records if r["status"] == "eligible"]
        for position, record in enumerate(eligible):
            if position == 0:
                record["status"], record["stage"] = "kept", "selected"
            else:
                record["status"], record["stage"] = "dropped", "selection"
                record["reasons"] = ["not_selected_one_question_per_image"]
                chosen = eligible[0]
                record["review_note"] = (
                    f"same answer as {chosen['question_id']}" if record["answer"] == chosen["answer"]
                    else f"answer {record['answer']} differs from kept {chosen['question_id']} ({chosen['answer']})"
                )
        records += image_records
    unused = set(review) - {r["question_id"] for r in records if r["stage"] == "agent_review"}
    if unused:
        raise ValueError(f"agent review entries match no rule-passing question: {sorted(unused)}")
    return records


def image_status(records: list[dict]) -> dict[str, dict]:
    """Image ID -> kept question or the image-level drop reason."""
    images: dict[str, dict] = {}
    for record in records:
        entry = images.setdefault(record["image_id"], {"row": record["row"], "kept": None, "reasons": Counter()})
        if record["status"] == "kept":
            entry["kept"] = record["question_id"]
        else:
            entry["reasons"].update(record["reasons"])
    out = {}
    for image_id, entry in images.items():
        if entry["kept"]:
            out[image_id] = {"row": entry["row"], "status": "kept", "question_id": entry["kept"]}
        else:
            primary = next(iter(entry["reasons"]))
            if primary not in ("not_focused", "duplicate_image_conflicting_answers", "duplicate_image"):
                primary = "no_eligible_question"
            out[image_id] = {"row": entry["row"], "status": "dropped", "reason": primary,
                             "question_reasons": dict(sorted(entry["reasons"].items()))}
    return out


# ---------------------------------------------------------------- loading
def load_rows(raw: Path, verify: bool = True) -> tuple[list[Row], dict]:
    """Read the parquet shards in name order. Row index is the position across shards."""
    import pyarrow.parquet as pq

    files = sorted((raw / "data").glob("test-*.parquet"))
    names = [str(path.relative_to(raw)) for path in files]
    if names != sorted(PARQUET_SHA256):
        raise ValueError(f"unexpected parquet files under {raw}: {names}")
    hashes = {}
    for path, name in zip(files, names):
        digest = file_sha256(path)
        if verify and digest != PARQUET_SHA256[name]:
            raise ValueError(f"{name}: sha256 {digest} differs from the pinned revision")
        hashes[name] = digest
    rows: list[Row] = []
    for path, name in zip(files, names):
        for record in pq.read_table(path).to_pylist():
            data = record["image"]["bytes"]
            with Image.open(io.BytesIO(data)) as image:
                if image.getexif().get(0x0112, 1) != 1:
                    raise ValueError(f"row {len(rows)} has an EXIF orientation; handle it before prep")
                width, height = image.size
                fmt, mode = image.format, image.mode
            config = json.loads(record["full_config"])
            if ([qa["question"] for qa in config["QA"]] != record["questions"]
                    or [qa["answer"] for qa in config["QA"]] != record["answers"]
                    or config["focused"] != record["is_focused"]):
                raise ValueError(f"row {len(rows)}: full_config disagrees with the columns")
            rows.append(Row(
                index=len(rows), sha256=hashlib.sha256(data).hexdigest(), image_bytes=data,
                image_format=fmt, mode=mode, width=width, height=height,
                questions=list(record["questions"]), answers=list(record["answers"]),
                objects=list(record["objects"]), categories=list(record["categories"]),
                is_focused=bool(record["is_focused"]), shard=name,
            ))
    source = {
        "repo": REPO, "revision": REVISION, "license": LICENSE, "citation": CITATION,
        "parquet_sha256": hashes,
        "readme_sha256": file_sha256(raw / "README.md") if (raw / "README.md").exists() else None,
    }
    return rows, source


def dataset_summary(rows: list[Row]) -> dict:
    sha_counts = Counter(row.sha256 for row in rows)
    return {
        "images": len(rows),
        "questions": sum(len(row.questions) for row in rows),
        "focused_images": sum(row.is_focused for row in rows),
        "focused_questions": sum(len(row.questions) for row in rows if row.is_focused),
        "duplicate_image_rows": sorted(row.image_id for row in rows if sha_counts[row.sha256] > 1),
        "formats": dict(Counter(row.image_format for row in rows)),
        "modes": dict(Counter(row.mode for row in rows)),
        "exif_orientation_rows": 0,
    }


# ---------------------------------------------------------------- writing eligibility
TSV_COLUMNS = (
    "question_id", "image_id", "row", "question_index", "status", "stage", "reasons", "answer",
    "target", "question", "is_focused", "width", "height", "image_sha256", "review_note",
)


def eligibility_tsv(records: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(TSV_COLUMNS)
    for r in records:
        writer.writerow([
            r["question_id"], r["image_id"], r["row"], r["question_index"], r["status"], r["stage"],
            ",".join(r["reasons"]), r["answer"], r["target"], r["question"],
            str(r["is_focused"]).lower(), r["width"], r["height"], r["image_sha256"], r["review_note"],
        ])
    return buffer.getvalue()


def eligibility_payload(records: list[dict], rows: list[Row], source: dict, tsv_sha256: str) -> dict:
    images = image_status(records)
    kept = [r for r in records if r["status"] == "kept"]
    question_hist = Counter(reason for r in records if r["status"] == "dropped" for reason in r["reasons"])
    primary_hist = Counter(r["reasons"][0] for r in records if r["status"] == "dropped")
    focused_hist = Counter(
        reason for r in records if r["status"] == "dropped" and r["is_focused"] for reason in r["reasons"]
    )
    image_hist = Counter(entry["reason"] for entry in images.values() if entry["status"] == "dropped")
    bins = Counter(counting.count_bin(r["answer_int"]) for r in kept)
    return {
        "protocol": counting.PROTOCOL,
        "rules_version": RULES_VERSION,
        "source": source,
        "dataset": dataset_summary(rows),
        "id_scheme": "countqa_<row index across sorted parquet shards, 4 digits>_<sha256(image bytes)[:8]>; "
        "question rows append _q<k> (index in the release questions list)",
        "selection": "is_focused rows only; one target object and one question per image; deterministic "
        "rules first, then the recorded agent review (question text only, not a human review); the first "
        "surviving question in release order is kept",
        "reasons": REASONS,
        "agent_review": {
            "reviewer": "agent review (Claude, question text only; no images viewed; not human review)",
            "entries": {r["question_id"]: {"reason": r["reasons"][0], "note": r["review_note"]}
                        for r in records if r["stage"] == "agent_review"},
        },
        "counts": {
            "kept_images": len(kept),
            "dropped_images": sum(entry["status"] == "dropped" for entry in images.values()),
            "kept_questions": len(kept),
            "dropped_questions": sum(r["status"] == "dropped" for r in records),
            "kept_count_bins": dict(sorted(bins.items())),
            "kept_answer_min": min(r["answer_int"] for r in kept) if kept else None,
            "kept_answer_max": max(r["answer_int"] for r in kept) if kept else None,
        },
        "histograms": {
            "dropped_images_by_reason": dict(image_hist.most_common()),
            "dropped_questions_by_primary_reason": dict(primary_hist.most_common()),
            "dropped_questions_by_any_reason": dict(question_hist.most_common()),
            "dropped_focused_questions_by_any_reason": dict(focused_hist.most_common()),
        },
        "kept": {r["image_id"]: {"question_id": r["question_id"], "question": r["question"],
                                 "target": r["target"], "answer": r["answer_int"],
                                 "width": r["width"], "height": r["height"],
                                 "image_sha256": r["image_sha256"]} for r in kept},
        "dropped_images": {image_id: {"reason": entry["reason"], "question_reasons": entry["question_reasons"]}
                           for image_id, entry in sorted(images.items()) if entry["status"] == "dropped"},
        "eligibility_tsv_sha256": tsv_sha256,
    }


def write_frozen(out: Path, records: list[dict], rows: list[Row], source: dict) -> dict:
    """Write eligibility.tsv/json and eligibility.sha256; refuse to change a frozen version."""
    out.mkdir(parents=True, exist_ok=True)
    tsv = eligibility_tsv(records)
    tsv_sha = hashlib.sha256(tsv.encode()).hexdigest()
    payload = eligibility_payload(records, rows, source, tsv_sha)
    json_path, tsv_path, sha_path = out / "eligibility.json", out / "eligibility.tsv", out / "eligibility.sha256"
    if json_path.exists():
        frozen = json.loads(json_path.read_text())
        frozen_at = frozen.pop("frozen_at", None)
        frozen.pop("frozen_before_model_outputs", None)
        if frozen != payload or tsv_path.read_text() != tsv:
            raise SystemExit(
                f"{json_path} is frozen (at {frozen_at}) and this run would change it; "
                "eligibility must not change after model outputs exist"
            )
        print(f"eligibility unchanged since {frozen_at}")
        return frozen | {"frozen_at": frozen_at}
    payload["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["frozen_before_model_outputs"] = True
    tsv_path.write_text(tsv)
    json_path.write_text(json.dumps(payload, indent=1) + "\n")
    sha_path.write_text(
        f"{file_sha256(tsv_path)}  eligibility.tsv\n{file_sha256(json_path)}  eligibility.json\n"
    )
    return payload


# ---------------------------------------------------------------- letterbox prep
def to_rgb(image: Image.Image) -> tuple[Image.Image, str]:
    """RGB pixels and how they were obtained. Transparent pixels go over LETTERBOX_FILL."""
    if image.mode == "RGB":
        return image, "none"
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        if rgba.getchannel("A").getextrema() == (255, 255):
            return rgba.convert("RGB"), "alpha_opaque_dropped"
        base = Image.new("RGBA", rgba.size, (*counting.LETTERBOX_FILL, 255))
        return Image.alpha_composite(base, rgba).convert("RGB"), "alpha_composited_over_fill"
    return image.convert("RGB"), f"converted_from_{image.mode}"


def letterbox_image(image: Image.Image) -> tuple[Image.Image, counting.SquareFrame, str]:
    rgb, alpha = to_rgb(image)
    frame = counting.letterbox_frame(*rgb.size)
    canvas = Image.new("RGB", (frame.side, frame.side), counting.LETTERBOX_FILL)
    canvas.paste(rgb, (-frame.left, -frame.top))
    return canvas, frame, alpha


def _write_one(job: tuple[str, bytes, str]) -> tuple[str, dict]:
    image_id, data, out_dir = job
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        canvas, frame, alpha = letterbox_image(image)
    path = Path(out_dir) / f"{image_id}.png"
    canvas.save(path, format="PNG", compress_level=6)
    return image_id, {"frame": frame.as_dict(), "alpha": alpha, "png_sha256": file_sha256(path)}


def prepare_letterbox(rows: list[Row], eligibility: dict, source: dict, work: Path, workers: int) -> dict:
    out_dir = work / "data" / "countqa" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {row.image_id: row for row in rows}
    kept = eligibility["kept"]
    stale = {p.stem for p in out_dir.glob("*.png")} - set(kept)
    if stale:
        raise SystemExit(f"{out_dir} holds images outside the frozen eligibility: {sorted(stale)[:5]}")
    jobs = [(image_id, by_id[image_id].image_bytes, str(out_dir)) for image_id in sorted(kept)]
    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (image_id, info) in enumerate(pool.map(_write_one, jobs), 1):
            results[image_id] = info
            if done % 50 == 0 or done == len(jobs):
                print(f"letterbox {done}/{len(jobs)}", flush=True)
    images = {}
    for image_id in sorted(kept):
        row, entry, info = by_id[image_id], kept[image_id], results[image_id]
        frame = counting.letterbox_frame(row.width, row.height)
        if info["frame"] != frame.as_dict():
            raise AssertionError(f"{image_id}: frame mismatch")
        images[image_id] = {
            "row": row.index, "shard": row.shard, "source_sha256": row.sha256,
            "source_format": row.image_format, "source_mode": row.mode, "alpha": info["alpha"],
            "width": row.width, "height": row.height, "frame": info["frame"], "valid": list(frame.valid),
            "png": f"images/{image_id}.png", "png_sha256": info["png_sha256"],
            "question_id": entry["question_id"], "question": entry["question"], "target": entry["target"],
            "answer": entry["answer"], "bin": counting.count_bin(entry["answer"]),
        }
    manifest = {
        "protocol": counting.PROTOCOL,
        "dataset": "countqa",
        "source": source,
        "eligibility_sha256": file_sha256(RESULTS_DIR / "eligibility.json"),
        "eligibility_tsv_sha256": eligibility["eligibility_tsv_sha256"],
        "frame": "letterbox",
        "letterbox_fill": list(counting.LETTERBOX_FILL),
        "target_grid": counting.TARGET_GRID,
        "transfer_preprocessing": TRANSFER_PREPROCESSING,
        "images": images,
        "counts": {"images": len(images), "bins": dict(Counter(e["bin"] for e in images.values()))},
    }
    path = work / "data" / "countqa" / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(images)} letterbox PNGs to {out_dir} and {path}")
    return manifest


# ---------------------------------------------------------------- CLI
def download(raw: Path) -> None:
    from huggingface_hub import snapshot_download

    raw.mkdir(parents=True, exist_ok=True)
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=raw, max_workers=4)
    for name, expected in PARQUET_SHA256.items():
        digest = file_sha256(raw / name)
        if digest != expected:
            raise SystemExit(f"{name}: sha256 {digest} != {expected}")
    (raw / "source.json").write_text(json.dumps(
        {"repo": REPO, "revision": REVISION, "parquet_sha256": PARQUET_SHA256}, indent=1) + "\n")
    print(f"verified {len(PARQUET_SHA256)} parquet files at {REVISION}")


def default_work() -> Path:
    return Path(os.environ.get("VTB_WORK", Path.home() / "vtb-work" / "counting-v3"))


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("download", "eligibility", "letterbox", "all"))
    ap.add_argument("--work", type=Path, default=default_work())
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    raw = args.work / "data" / "countqa" / "raw"
    if args.step in ("download", "all"):
        download(raw)
    if args.step in ("eligibility", "letterbox", "all"):
        rows, source = load_rows(raw)
        summary = dataset_summary(rows)
        print(json.dumps(summary, indent=1))
        records = decide(rows)
        if args.step in ("eligibility", "all"):
            payload = write_frozen(args.results, records, rows, source)
            print(json.dumps({"counts": payload["counts"], "histograms": payload["histograms"]}, indent=1))
        if args.step in ("letterbox", "all"):
            eligibility = json.loads((args.results / "eligibility.json").read_text())
            if {k: v["question_id"] for k, v in eligibility["kept"].items()} != {
                r["image_id"]: r["question_id"] for r in records if r["status"] == "kept"
            }:
                raise SystemExit("frozen eligibility differs from the current rules")
            prepare_letterbox(rows, eligibility, source, args.work, args.workers)


if __name__ == "__main__":
    main()
