# CountQA exemplar boxes: annotation request

Status: **pending**. No human has drawn or approved a box. `../exemplars.json` has
`"status": "pending"`, and the human-box protocol of `scripts/countqa_eval.py` will not run the
exemplar-conditioned arm until a human finishes these steps.

Separately, AI-authored boxes are in `../exemplars_ai.json`. No human has reviewed them. They
feed only the experimental protocol `countqa-ai-exemplars-v1` (`countqa_eval.py
--annotation-source ai`), described in `../README.md`. They are not approved boxes and do not
satisfy this request.

We need one box per kept question: 439 questions, one per image, all from the frozen subset in
`../eligibility.json`. Boxing takes about 5 to 10 seconds per image, so plan on roughly an hour.

## Protocol

1. Read the question at the top of the page (for example "How many limes are there?"). The
   object to box appears after **box one:**.
2. Pick **one** instance of that object that is clearly visible and not occluded. Pick a typical
   one, not the biggest or the oddest.
3. Drag a **tight** box around the instance's visible extent. Don't box a group, a part of the
   object, a reflection, a printed picture of the object, or a partly hidden instance.
4. If the image has no clean instance (every instance is occluded, cut off, too blurry to
   delimit, or you can't tell what the question means), press **No clean instance** (`x`). The
   question is then excluded from the conditioned evaluation with reason `no_clean_instance`.
   Don't guess a box.
5. Draw exactly one box per question. Drawing again replaces the box.
6. Use only your own judgement. Nobody has supplied AI-suggested boxes. If a
   `suggested_unreviewed` field ever appears, it must never be copied into an approved box.

The tool records boxes in continuous source pixels of the released image, `(x0, y0, x1, y1)`.
`finalize` adds the box normalized by width and height and the box in letterbox unit coordinates.

## Steps

The tool is already built on the server at
`~/vtb-work/counting-v3/data/countqa/annotate/` (`index.html` plus `images/`, the 439 released
image files unchanged, about 245 MB). To rebuild it, run
`uv run python -m scripts.countqa_annotate html` from the repository with `VTB_WORK` set.

1. **Open the tool.** Use either option.
   - Copy it to your laptop and open it locally:
     ```bash
     rsync -a <user>@<server>:vtb-work/counting-v3/data/countqa/annotate/ ~/countqa-annotate/
     open ~/countqa-annotate/index.html      # or double-click it; Chrome or Firefox
     ```
   - Or serve it from the server through an SSH tunnel:
     ```bash
     ssh -L 8765:127.0.0.1:8765 <user>@<server> \
       'python3 -m http.server --bind 127.0.0.1 --directory ~/vtb-work/counting-v3/data/countqa/annotate 8765'
     # then open http://localhost:8765/ in your browser
     ```
2. Type your name in **Annotator**.
3. For each question, draw the box and press **Enter** (Save box), or press **x** (No clean
   instance). The tool jumps to the next pending question. Other keys: `n`/`p` next and previous,
   `g` next pending, `c` clear, plus a zoom selector (fit, 100%, 200%) for small objects.
   Progress is saved in the browser automatically. It's tied to the browser and to the URL
   (file vs localhost), so keep using the same one.
4. Press **Export JSON** whenever you stop, and again at the end. Each export is a full snapshot.
   **Import** reloads an export to resume, for example in another browser.
5. When the progress bar shows `pending 0`, copy the final export to the server:
   ```bash
   scp countqa_exemplars_<name>_<time>.json <user>@<server>:vtb-work/counting-v3/data/countqa/annotations/
   ```
6. Finalize. A human runs this; it records who approved the boxes:
   ```bash
   cd ~/vision-tower-bench
   export VTB_WORK=$HOME/vtb-work/counting-v3
   uv run python -m scripts.countqa_annotate finalize \
     --export $VTB_WORK/data/countqa/annotations/countqa_exemplars_<name>_<time>.json \
     --approved-by "<your name>"
   ```
   `finalize` refuses an export with pending questions, an export from a different sheet or
   eligibility, a missing annotator name, or a box outside the image or smaller than 2 px. It
   writes `results/counting/countqa/exemplars.json` with `"status": "approved"` and the sha256 of
   your export. Commit that file. The raw export stays under `$VTB_WORK`.

## Files

| file | what |
|---|---|
| `sheet.csv` | the request: stable question and image IDs, question, target object, answer, image size, empty box fields (pixels and normalized), `status = pending` |
| `../exemplars.json` | the pending manifest now, and the approved boxes after `finalize` |
| `../eligibility.{tsv,json,sha256}` | frozen subset; kept and dropped questions with reason codes |
| `scripts/countqa_annotate.py` | `sheet`, `html` and `finalize` |
