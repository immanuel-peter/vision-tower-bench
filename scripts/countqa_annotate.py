#!/usr/bin/env python3
"""Human exemplar-box annotation for the frozen CountQA subset.

Steps (see results/counting/countqa/annotation/README.md for the human instructions):

``sheet``     write the pending annotation sheet (CSV) and the pending exemplars.json from the
              frozen eligibility. No box is filled in and every status is "pending".
``html``      write a self-contained local tool, $VTB_WORK/data/countqa/annotate/index.html, next to
              the original released image files, for drawing one box per kept question.
``finalize``  validate the tool's JSON export against the sheet and write the approved
              results/counting/countqa/exemplars.json that countqa_eval.py reads.

Boxes are stored in continuous source pixels of the released image, (x0, y0, x1, y1) with
pixel (row i, col j) covering [j, j+1) x [i, i+1), plus the same box normalized by the image
width and height, plus the box in letterbox unit coordinates (counting.letterbox_frame).
AI-suggested boxes may only ever appear in a separate ``suggested_unreviewed`` field and are
never copied into an approved box.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vtb import counting

RESULTS = Path("results/counting/countqa")
SHEET = RESULTS / "annotation" / "sheet.csv"
EXEMPLARS = RESULTS / "exemplars.json"
EXPORT_FORMAT = "vtb-countqa-annotation-export-v1"
EXEMPLAR_FORMAT = "vtb-countqa-exemplars-v1"
MIN_BOX_PX = 2.0
PROTOCOL = (
    "One exemplar box per kept question, drawn by a human on the released image. Box one clearly "
    "visible, unoccluded instance of the object the question asks about, tight to its visible "
    "extent. Never box a group, a part, a reflection or a picture of the object. If no clean "
    "instance exists, mark no_clean_instance; the question is then excluded from the conditioned "
    "evaluation with that reason. AI-suggested boxes are never approved boxes."
)
SHEET_COLUMNS = (
    "question_id", "image_id", "question", "target", "answer", "width", "height",
    "box_x0_px", "box_y0_px", "box_x1_px", "box_y1_px",
    "box_x0_unit", "box_y0_unit", "box_x1_unit", "box_y1_unit",
    "status", "annotator", "annotated_at", "note", "suggested_unreviewed",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def load_eligibility(results: Path = RESULTS) -> tuple[dict, str]:
    path = results / "eligibility.json"
    return json.loads(path.read_text()), sha256_file(path)


def sheet_rows(eligibility: dict) -> list[dict]:
    rows = []
    for image_id, entry in sorted(eligibility["kept"].items()):
        rows.append({
            "question_id": entry["question_id"], "image_id": image_id, "question": entry["question"],
            "target": entry["target"], "answer": entry["answer"], "width": entry["width"],
            "height": entry["height"], **{column: "" for column in SHEET_COLUMNS[7:15]},
            "status": "pending", "annotator": "", "annotated_at": "", "note": "", "suggested_unreviewed": "",
        })
    return rows


def sheet_csv(rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, SHEET_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def pending_exemplars(eligibility: dict, eligibility_sha: str, sheet_sha: str) -> dict:
    items = {}
    for image_id, entry in sorted(eligibility["kept"].items()):
        items[entry["question_id"]] = {
            "image_id": image_id, "status": "pending", "box_px": None, "box_source_unit": None,
            "box_letterbox_unit": None, "box_source": None, "annotated_at": None, "note": "",
            "suggested_unreviewed": None,
        }
    return {
        "format": EXEMPLAR_FORMAT, "status": "pending", "protocol": PROTOCOL,
        "coordinates": "box_px: continuous source pixels (x0, y0, x1, y1) of the released image; "
        "box_source_unit: box_px / (w, h, w, h); box_letterbox_unit: counting.letterbox_frame(w, h).boxes",
        "eligibility_sha256": eligibility_sha, "sheet_sha256": sheet_sha,
        "annotator": None, "approved_by": None, "approved_at": None, "export_sha256": None,
        "counts": {"questions": len(items), "approved": 0, "no_clean_instance": 0, "pending": len(items)},
        "items": items,
    }


def validate_box(box, width: int, height: int) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError(f"box must be four finite numbers, got {box}")
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(f"box {box.tolist()} leaves the {width}x{height} image")
    if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
        raise ValueError(f"box {box.tolist()} is smaller than {MIN_BOX_PX} px")
    return box


def finalize(export: dict, export_sha: str, eligibility: dict, eligibility_sha: str, sheet_sha: str,
             approved_by: str, now: str | None = None) -> dict:
    """Validate a tool export and return the approved exemplars payload."""
    if export.get("format") != EXPORT_FORMAT:
        raise ValueError(f"not a {EXPORT_FORMAT} export")
    if export.get("sheet_sha256") != sheet_sha or export.get("eligibility_sha256") != eligibility_sha:
        raise ValueError("export was made from a different sheet or eligibility")
    annotator = (export.get("annotator") or "").strip()
    if not annotator or not approved_by.strip():
        raise ValueError("the export needs an annotator name and finalize needs --approved-by")
    kept = {entry["question_id"]: (image_id, entry) for image_id, entry in eligibility["kept"].items()}
    seen = [item["question_id"] for item in export["items"]]
    if len(seen) != len(set(seen)) or set(seen) != set(kept):
        raise ValueError("export items do not match the kept questions one to one")
    items = {}
    for item in sorted(export["items"], key=lambda item: item["question_id"]):
        image_id, entry = kept[item["question_id"]]
        if item.get("image_id") != image_id:
            raise ValueError(f"{item['question_id']}: image ID mismatch")
        status = item.get("status")
        record = {"image_id": image_id, "box_px": None, "box_source_unit": None, "box_letterbox_unit": None,
                  "box_source": None, "annotated_at": item.get("annotated_at"), "note": item.get("note", ""),
                  "suggested_unreviewed": item.get("suggested_unreviewed")}
        if status == "boxed":
            box = validate_box(item["box_px"], entry["width"], entry["height"])
            frame = counting.letterbox_frame(entry["width"], entry["height"])
            letterbox, kept_fraction = counting.clip_box(frame.boxes(box)[0])
            if kept_fraction < 1.0 - 1e-9:
                raise AssertionError(f"{item['question_id']}: a source box must lie inside the letterbox canvas")
            size = np.array([entry["width"], entry["height"]] * 2, dtype=np.float64)
            record |= {"status": "approved", "box_px": [round(float(v), 2) for v in box],
                       "box_source_unit": [round(float(v), 6) for v in box / size],
                       "box_letterbox_unit": [round(float(v), 6) for v in letterbox], "box_source": "human"}
        elif status == "no_clean_instance":
            record["status"] = "no_clean_instance"
        else:
            raise ValueError(f"{item['question_id']} is still {status!r}; finish every question first")
        items[item["question_id"]] = record
    statuses = [record["status"] for record in items.values()]
    return {
        "format": EXEMPLAR_FORMAT, "status": "approved", "protocol": PROTOCOL,
        "coordinates": pending_exemplars(eligibility, eligibility_sha, sheet_sha)["coordinates"],
        "eligibility_sha256": eligibility_sha, "sheet_sha256": sheet_sha, "annotator": annotator,
        "exported_at": export.get("exported_at"), "approved_by": approved_by.strip(),
        "approved_at": now or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "export_sha256": export_sha,
        "counts": {"questions": len(items), "approved": statuses.count("approved"),
                   "no_clean_instance": statuses.count("no_clean_instance"), "pending": 0},
        "items": items,
    }


# ---------------------------------------------------------------- HTML tool
HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>CountQA exemplar boxes</title>
<style>
 body{font:14px/1.4 system-ui,sans-serif;margin:0;background:#f4f4f2;color:#222}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccc;padding:8px 12px;z-index:2;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
 header input[type=text]{width:12em}
 button{font:inherit;padding:4px 10px;cursor:pointer}
 button.primary{background:#1f5fbf;color:#fff;border:1px solid #1f5fbf}
 button.warn{background:#b5521b;color:#fff;border:1px solid #b5521b}
 main{padding:10px 12px}
 #q{font-size:20px;font-weight:600}
 #meta{color:#555;margin:2px 0 8px}
 #wrap{position:relative;display:inline-block;background:rgb(124,116,104);overflow:auto;max-width:100%;max-height:78vh;border:1px solid #999}
 #img{display:block;user-select:none;-webkit-user-drag:none}
 #box{position:absolute;border:2px solid #ff1f8f;box-shadow:0 0 0 1px #fff;pointer-events:none;display:none}
 #status{font-weight:600}
 .pending{color:#777}.boxed{color:#1a7f37}.no_clean_instance{color:#b5521b}
 details{margin:6px 0;max-width:70em}
 #err{color:#b00;font-weight:600}
</style></head><body>
<header>
 <label>Annotator <input id="annotator" type="text" placeholder="your name"></label>
 <button id="prev">&larr; Prev (p)</button><button id="next">Next (n) &rarr;</button>
 <button id="nextPending">Next pending (g)</button>
 <span id="progress"></span>
 <label>Zoom <select id="zoom"><option value="fit">fit</option><option value="1">100%</option><option value="2">200%</option></select></label>
 <button id="export" class="primary">Export JSON</button>
 <label>Import <input id="import" type="file" accept=".json"></label>
</header>
<main>
 <details><summary>Protocol (read once)</summary><div id="protocol"></div>
 <p>Drag on the image to draw the box; drawing again replaces it. <b>Enter</b> or <b>Save box</b> stores
 it and moves on. <b>x</b> marks <i>no clean instance</i>. <b>c</b> clears this question back to pending.
 Work is saved in this browser automatically; export the JSON when done (and as a backup).</p></details>
 <div id="q"></div>
 <div id="meta"></div>
 <div style="margin:6px 0"><button id="save" class="primary">Save box (Enter)</button>
 <button id="noclean" class="warn">No clean instance (x)</button>
 <button id="clear">Clear (c)</button>
 <input id="note" type="text" placeholder="optional note" style="width:24em">
 Status: <span id="status"></span> <span id="err"></span></div>
 <div id="wrap"><img id="img" draggable="false" alt=""><div id="box"></div></div>
</main>
<script>
const DATA = __DATA__;
const KEY = "vtb-countqa-annotation-" + DATA.sheet_sha256;
let state = JSON.parse(localStorage.getItem(KEY) || "null") || {annotator: "", items: {}};
let index = 0, draft = null, drag = null;
const $ = id => document.getElementById(id);
$("protocol").textContent = DATA.protocol;
$("annotator").value = state.annotator || "";
function item(){ return DATA.items[index]; }
function rec(q){ return state.items[q.question_id] || {status: "pending", box_px: null, note: ""}; }
function persist(){ state.annotator = $("annotator").value.trim(); localStorage.setItem(KEY, JSON.stringify(state)); progress(); }
function progress(){
  let boxed = 0, none = 0;
  for (const q of DATA.items){ const s = rec(q).status; if (s === "boxed") boxed++; else if (s === "no_clean_instance") none++; }
  $("progress").textContent = `${index + 1} / ${DATA.items.length}  ·  boxed ${boxed}  ·  no clean instance ${none}  ·  pending ${DATA.items.length - boxed - none}`;
}
function scale(){ const img = $("img"); return img.naturalWidth ? img.getBoundingClientRect().width / img.naturalWidth : 1; }
function drawBox(b){
  const el = $("box");
  if (!b){ el.style.display = "none"; return; }
  const s = scale();
  el.style.display = "block";
  el.style.left = (b[0] * s) + "px"; el.style.top = (b[1] * s) + "px";
  el.style.width = ((b[2] - b[0]) * s) + "px"; el.style.height = ((b[3] - b[1]) * s) + "px";
}
function applyZoom(){
  const img = $("img"), z = $("zoom").value;
  if (!img.naturalWidth) return;
  if (z === "fit"){
    const maxW = document.querySelector("main").clientWidth - 30, maxH = window.innerHeight * 0.76;
    const s = Math.min(1, maxW / img.naturalWidth, maxH / img.naturalHeight);
    img.style.width = (img.naturalWidth * s) + "px";
  } else { img.style.width = (img.naturalWidth * Number(z)) + "px"; }
  drawBox(draft);
}
function show(){
  const q = item(), r = rec(q);
  $("q").textContent = q.question;
  $("meta").textContent = `${q.question_id}  ·  box one: ${q.target}  ·  answer ${q.answer}  ·  ${q.width}×${q.height} px`;
  $("note").value = r.note || "";
  $("status").textContent = r.status; $("status").className = r.status; $("err").textContent = "";
  draft = r.box_px ? r.box_px.slice() : null;
  const img = $("img");
  img.onload = () => {
    if (img.naturalWidth !== q.width || img.naturalHeight !== q.height){
      $("err").textContent = `image is ${img.naturalWidth}×${img.naturalHeight}, expected ${q.width}×${q.height}`;
    }
    applyZoom();
  };
  img.src = q.src; drawBox(null); progress();
}
function toImage(ev){
  const r = $("img").getBoundingClientRect(), q = item();
  const x = (ev.clientX - r.left) / r.width * q.width, y = (ev.clientY - r.top) / r.height * q.height;
  return [Math.min(Math.max(x, 0), q.width), Math.min(Math.max(y, 0), q.height)];
}
$("img").addEventListener("mousedown", ev => { ev.preventDefault(); drag = toImage(ev); });
window.addEventListener("mousemove", ev => {
  if (!drag) return;
  const p = toImage(ev);
  draft = [Math.min(drag[0], p[0]), Math.min(drag[1], p[1]), Math.max(drag[0], p[0]), Math.max(drag[1], p[1])].map(v => Math.round(v * 10) / 10);
  drawBox(draft);
});
window.addEventListener("mouseup", () => { drag = null; });
function setRecord(status){
  const q = item();
  if (!$("annotator").value.trim()){ $("err").textContent = "enter your name first"; return false; }
  if (status === "boxed"){
    if (!draft || draft[2] - draft[0] < DATA.min_box_px || draft[3] - draft[1] < DATA.min_box_px){ $("err").textContent = "draw a box first"; return false; }
  }
  state.items[q.question_id] = {status, box_px: status === "boxed" ? draft.slice() : null,
    note: $("note").value.trim(), annotated_at: new Date().toISOString()};
  persist(); return true;
}
function go(i){ index = Math.max(0, Math.min(DATA.items.length - 1, i)); show(); }
function nextPending(){
  for (let k = 1; k <= DATA.items.length; k++){
    const i = (index + k) % DATA.items.length;
    if (rec(DATA.items[i]).status === "pending"){ go(i); return; }
  }
  $("err").textContent = "no pending questions left; export the JSON";
}
$("save").onclick = () => { if (setRecord("boxed")) nextPending(); };
$("noclean").onclick = () => { if (setRecord("no_clean_instance")) nextPending(); };
$("clear").onclick = () => { delete state.items[item().question_id]; persist(); show(); };
$("prev").onclick = () => go(index - 1);
$("next").onclick = () => go(index + 1);
$("nextPending").onclick = nextPending;
$("zoom").onchange = applyZoom;
$("annotator").onchange = persist;
window.addEventListener("resize", applyZoom);
document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "INPUT") { if (ev.key === "Enter" && ev.target.id === "note") $("save").click(); return; }
  if (ev.key === "Enter") $("save").click();
  else if (ev.key === "x") $("noclean").click();
  else if (ev.key === "c") $("clear").click();
  else if (ev.key === "n" || ev.key === "ArrowRight") go(index + 1);
  else if (ev.key === "p" || ev.key === "ArrowLeft") go(index - 1);
  else if (ev.key === "g") nextPending();
});
$("export").onclick = () => {
  persist();
  if (!state.annotator){ $("err").textContent = "enter your name first"; return; }
  const out = {format: DATA.export_format, sheet_sha256: DATA.sheet_sha256, eligibility_sha256: DATA.eligibility_sha256,
    annotator: state.annotator, exported_at: new Date().toISOString(), tool: "scripts/countqa_annotate.py html",
    items: DATA.items.map(q => { const r = rec(q); return {question_id: q.question_id, image_id: q.image_id,
      status: r.status, box_px: r.box_px, image_width: q.width, image_height: q.height,
      annotated_at: r.annotated_at || null, note: r.note || ""}; })};
  const blob = new Blob([JSON.stringify(out, null, 1)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `countqa_exemplars_${state.annotator.replace(/\W+/g, "_")}_${out.exported_at.replace(/[:.]/g, "-")}.json`;
  a.click();
};
$("import").onchange = ev => {
  const f = ev.target.files[0]; if (!f) return;
  f.text().then(t => {
    const d = JSON.parse(t);
    if (d.sheet_sha256 !== DATA.sheet_sha256){ $("err").textContent = "that export belongs to a different sheet"; return; }
    state.annotator = d.annotator || state.annotator; state.items = {};
    for (const it of d.items) if (it.status !== "pending") state.items[it.question_id] = {status: it.status, box_px: it.box_px, note: it.note || "", annotated_at: it.annotated_at};
    $("annotator").value = state.annotator; persist(); show();
  });
};
{ const first = DATA.items.findIndex(q => rec(q).status === "pending"); go(first < 0 ? 0 : first); }
</script></body></html>
"""


def tool_items(eligibility: dict, image_names: dict[str, str]) -> list[dict]:
    return [{"question_id": entry["question_id"], "image_id": image_id, "question": entry["question"],
             "target": entry["target"], "answer": entry["answer"], "width": entry["width"],
             "height": entry["height"], "src": f"images/{image_names[image_id]}"}
            for image_id, entry in sorted(eligibility["kept"].items())]


def render_html(items: list[dict], eligibility_sha: str, sheet_sha: str) -> str:
    data = {"items": items, "eligibility_sha256": eligibility_sha, "sheet_sha256": sheet_sha,
            "protocol": PROTOCOL, "export_format": EXPORT_FORMAT, "min_box_px": MIN_BOX_PX}
    return HTML.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))


def write_tool(work: Path, eligibility: dict, eligibility_sha: str, sheet_sha: str) -> Path:
    """Copy the released image bytes of kept rows and write index.html beside them."""
    from scripts import prep_countqa

    rows, _ = prep_countqa.load_rows(work / "data" / "countqa" / "raw")
    by_id = {row.image_id: row for row in rows}
    out = work / "data" / "countqa" / "annotate"
    (out / "images").mkdir(parents=True, exist_ok=True)
    names = {}
    for image_id, entry in sorted(eligibility["kept"].items()):
        row = by_id[image_id]
        if row.sha256 != entry["image_sha256"]:
            raise ValueError(f"{image_id}: image bytes differ from the frozen eligibility")
        names[image_id] = f"{image_id}.{'png' if row.image_format == 'PNG' else 'jpg'}"
        path = out / "images" / names[image_id]
        if not path.exists() or sha256_file(path) != row.sha256:
            path.write_bytes(row.image_bytes)
    html = out / "index.html"
    html.write_text(render_html(tool_items(eligibility, names), eligibility_sha, sheet_sha))
    return html


def default_work() -> Path:
    return Path(os.environ.get("VTB_WORK", Path.home() / "vtb-work" / "counting-v3"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("sheet", "html", "finalize"))
    ap.add_argument("--work", type=Path, default=default_work())
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--export", type=Path, help="JSON exported by the HTML tool (finalize)")
    ap.add_argument("--approved-by", default="", help="name of the human approving the boxes (finalize)")
    args = ap.parse_args(argv)
    eligibility, eligibility_sha = load_eligibility(args.results)
    sheet_path = args.results / "annotation" / "sheet.csv"
    exemplars_path = args.results / "exemplars.json"
    if args.step == "sheet":
        text = sheet_csv(sheet_rows(eligibility))
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet_path.write_text(text)
        if exemplars_path.exists() and json.loads(exemplars_path.read_text())["status"] != "pending":
            raise SystemExit(f"{exemplars_path} already holds approved boxes; not overwriting")
        payload = pending_exemplars(eligibility, eligibility_sha, sha256_file(sheet_path))
        exemplars_path.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {sheet_path} and pending {exemplars_path} ({len(payload['items'])} questions)")
    elif args.step == "html":
        html = write_tool(args.work, eligibility, eligibility_sha, sha256_file(sheet_path))
        print(f"wrote {html}")
    else:
        if args.export is None:
            raise SystemExit("finalize needs --export")
        data = args.export.read_bytes()
        payload = finalize(json.loads(data), sha256_bytes(data), eligibility, eligibility_sha,
                           sha256_file(sheet_path), args.approved_by)
        exemplars_path.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote approved {exemplars_path}: {json.dumps(payload['counts'])}")


if __name__ == "__main__":
    main()
