# CountQA: curated transfer subset for the FSC-147 count readouts

**Experimental and supplementary.** This is exemplar-based frozen-feature counting transfer on a
curated CountQA-derived subset, read by FSC-147-trained heads. These are not official CountQA
benchmark scores, and they are not MLLM question answering. Don't put these numbers next to
published CountQA MLLM scores. FSC-147 is the primary counting experiment. CountQA adds exploratory
transfer evidence and should not by itself support headline claims about Tower quality.

Source: `Jayant-Sravan/CountQA` at revision `f92cc6fe46542c61e2916e3d2ae9a911e2216b1a`,
CC-BY-4.0. Citation: Tamarapalli, Grover, Pande, Yerramilli, *CountQA: How Well Do MLLMs
Count in the Wild?*, arXiv:2508.06585 (2025). Images aren't committed. Rebuild them with
`scripts/prep_countqa.py`.

## Subset (frozen 2026-10-21, before any model output)

- 1,001 images, 1,528 questions and 524 `is_focused` images (677 questions), as the release
  states. All 13 parquet shards match the Hub's sha256 for this revision.
- Image ID: `countqa_<row>_<sha8>`. `row` is the position across the parquet shards sorted by
  name, and `sha8` is the first 8 hex digits of the sha256 of the image bytes. A question ID
  appends `_q<k>`. No IDs or prefixes collide.
- Rows 3 and 36 hold the same image bytes but answer "How many pipes" with 10 and 14, so both are
  dropped.
- Kept: **439 images, one question each**. Count bins on the answer: 435 `le50`, 3 `50to200`,
  1 `gt200`. Answers range from 1 to 400.
- Order of decisions:
  1. Deterministic rules on the question text drop colour, location, part-of, attribute
     (material, size, orientation or shape), combination, comparison or negation wording,
     generic targets, unparsed forms and zero answers.
  2. A recorded agent review of the question text only drops subtypes (card suits, brands,
     towel kinds), mixed categories (fruits, cutlery) and one restated part-of question
     (26 questions). This review is not a human review.
  3. The first surviving question in release order is kept.
- `eligibility.tsv`: every question, kept or dropped, with its stage and reason codes.
  `eligibility.json`: source hashes, rules, counts, histograms, kept and dropped image IDs.
  `eligibility.sha256` holds both files' hashes. `scripts/prep_countqa.py eligibility`
  refuses to change a frozen file.

## Preprocessing for transfer, compared with FSC-147

FSC-147 heads train on the central min-side square (`counting.centre_crop_frame`). CountQA
answers count the whole image, so each kept image is centred whole on a max-side square
(`counting.letterbox_frame`). The padding is filled with `counting.LETTERBOX_FILL`, and the
square is written as a PNG at native resolution
(`$VTB_WORK/data/countqa/images/<image_id>.png`, manifest `$VTB_WORK/data/countqa/manifest.json`).
The adapter then resizes that square to the benchmark resolution. The heads therefore see
padding they never saw in training, and objects at a smaller scale. Predicted density is summed
only over the valid area, `masked_count(density, coverage(frame.valid, 64))`. Exemplar boxes go
through `frame.boxes` and are never centre-cropped. Prototype centring uses
`token_valid(frame.valid, side, grid_extent)` at the token grid. Five kept PNG sources have a fully opaque alpha
channel, which is dropped. No kept image needed compositing, and no image has an EXIF
orientation.

## Exemplars

`exemplars.json` is **pending**. See `annotation/README.md` for the human annotation request,
the protocol and the local tool. No human box has been approved, so the human-box protocol below
has no results.

### AI-authored exemplars (experimental protocol `countqa-ai-exemplars-v1`)

`exemplars_ai.json` holds user-authorized, AI-authored boxes. Three `gpt-6-astra` instances drew
them through Codex/T3 (annotation protocol `user-authorized-ai-visual-exemplars-v1`, revision 2,
export sha256 `3c113eae…b382`). **No human reviewed them.** `annotation_source = "ai"`,
`human_reviewed = false` and `approved_by = null` are set on the file and on every row, and
`box_source = "ai"` on every boxed row. Excluded rows carry no box. The file status is
`ai_unreviewed`. Each AI-exemplar result repeats this provenance in its `annotation` block. The human finalizer (`countqa_annotate.py finalize`) was
not run on this export, because it would record the boxes as human-approved. The human loader in
`countqa_eval.py` also refuses this file.

`scripts/countqa_ai_exemplars.py` builds the file from the delivered package. It checks:

- the package `SHA256SUMS.txt`
- the frozen sheet and eligibility hashes
- every source image's sha256 against the prepared manifest
- image sizes
- box bounds, at least 2 px per side
- that each normalized box matches its pixel box
- the status counts

Coverage of the 439 frozen questions:

| status | questions | use |
|---|---:|---|
| `boxed` | 325 | included; one AI box each |
| `no_clean_instance` | 105 | excluded: no complete, unoccluded instance |
| `needs_review` | 9 | excluded: unresolved target identity or boundary; never boxed |

Review coverage:

- AI annotators inspected all 439 images.
- Every final box received an AI overlay recheck.
- An independent AI adjudicator reviewed 26 items.
- A parent AI spot-checked 24 items. `review_coverage.parent_ai_spot_check` copies those
  records verbatim, so their statuses are from before adjudication. Seven differ from the final file: six
  became `no_clean_instance` and one became `boxed`. `statement` is copied verbatim from the package audit.

None of this is human review. Count answers were not used to choose boxes or exclusions.

`countqa_eval.py --annotation-source ai` scores both arms on exactly the 325 AI-boxed image IDs.
Conditioned and exemplar-free cells therefore pair image for image. Results go to `ai_exemplars/`.
Each result records the annotation provenance, the full 439 → 325 breakdown, the sha256 of the
included IDs, and this label. The exemplar-free score on all 439 questions comes from the default
mode and stays a separate result.

Limitations:

- Transfer: heads trained on FSC-147 centre crops read letterboxed whole images.
- Annotation: boxes may contain AI errors in boundary or target identity.
- Selection: requiring a clean exemplar may remove the hardest images.
- Robustness: sensitivity to the choice of exemplar, and whether rankings hold under other
  exemplars or protocol choices, are not established.
- Calibration: no CountQA result selected any weight, rate, calibration or checkpoint.

## Evaluation (`scripts/countqa_eval.py`)

```bash
uv run python -m scripts.countqa_eval --arm exemplar --variant matched \
  --features $VTB_WORK/features/countqa/<model>_<res>_full \
  --readouts $VTB_WORK/readouts/count-v4/<model> --seeds 0 1 2
```

The script loads each `<arm>_<variant>/<stage>_L<layer>/` readout through
`vtb.geometry_run.load_readout(dir, seed, device)` and applies it frozen. It checks that each
readout's state hash is the same before and after scoring, and it never fits anything on
CountQA. Outputs:

- `results/counting/countqa/<model>_count_<exemplar|free>_<raw|matched>.json`
- a `.predictions.tsv` file with the contract columns and `split = countqa`

| | primary | secondary |
|---|---|---|
| Arm | exemplar-conditioned, one approved human box | exemplar-free |
| Stages | Tower | every Stage |
| Images | approved boxes only; `no_clean_instance` excluded | all 439; also reported on the exemplar subset once boxes are approved |

The conditioned arm refuses to run unless `exemplars.json` is `approved`.

Metrics:

- MAE, and RMSE as the mean over seeds of each seed's RMSE (`rmse`, the tables' RMSE ± sd);
  `rmse_pooled` is the square root of the mean `sq_error` over every seed × image row
- exact match, where a prediction counts as correct when `floor(pred + 0.5)` equals the answer
- per-bin values and sample counts (`counting.count_bin` of the answer)

Each payload's own `stage_differences` use `vtb.geometry_bootstrap.paired_metric_bootstrap`.
The committed intervals under `../bootstrap/` come from `scripts/count_bootstrap.py`
(`vtb.count_bootstrap`), which also resamples RMSE. Both pair identical image IDs and seed k
with seed k, average each image's metric over seeds, draw 10,000 resamples with bootstrap seed 0
and report a 95% percentile interval.

## Results (experimental, supplementary)

Every result in this section is experimental and supplementary. None is an official
CountQA benchmark score.

All 16 Towers have results:

- Exemplar-free on all 439 questions, at every final Stage: `<model>_count_free_<raw|matched>.json`.
  Paired Stage and capacity intervals are in `../bootstrap/countqa/`. Per-bin rows are
  in `../tables/` (435 of 439 answers fall in `le50`).
- AI-exemplar protocol on the 325 AI-boxed questions, both arms:
  `ai_exemplars/<model>_count_<arm>_<raw|matched>.json`. Paired intervals, including
  conditioned versus exemplar-free on identical IDs, are in
  `../bootstrap/ai_exemplars/countqa/`. The table is `ai_exemplars/count_table.md`.

On the 325 AI-boxed questions, the AI exemplar never clearly lowers the Tower's error
under 512-dim reduction. There the exemplar-free arm is better for 5 Towers and 11
are unresolved. With full-width features it is better for 5 Towers, the exemplar for
1, and 10 are unresolved.

This adds to the FSC-147 finding that the one-channel exemplar conditioning usually does
not help. It cannot separate a weak conditioning design from AI box errors or the
transfer gap.

Regenerate the CountQA intervals and the AI-exemplar table from the repository root:

```bash
uv run python -m scripts.count_bootstrap countqa results/counting/countqa/*_count_free_*.json \
  --out results/counting/bootstrap --relative-to . --label "<label>"
uv run python -m scripts.count_bootstrap countqa results/counting/countqa/ai_exemplars/*_count_*.json \
  --out results/counting/bootstrap/ai_exemplars --relative-to . --label "<label>"
uv run python -m scripts.count_table --results /nonexistent \
  --countqa results/counting/countqa/ai_exemplars/*_count_*.json \
  --out results/counting/countqa/ai_exemplars --label "<label>"
```

The committed outputs record their exact labels.

