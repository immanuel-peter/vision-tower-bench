# CountQA: curated transfer subset for count-v1

This is exemplar-based frozen-feature counting on a curated CountQA subset. It is not MLLM
question answering. Don't put these numbers next to MLLM CountQA scores.

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
`coverage(frame.valid, side)` at the token grid. Five kept PNG sources have a fully opaque alpha
channel, which is dropped. No kept image needed compositing, and no image has an EXIF
orientation.

## Exemplars

`exemplars.json` is **pending**. See `annotation/README.md` for the human annotation request,
the protocol and the local tool.

## Evaluation (`scripts/countqa_eval.py`)

```bash
uv run python -m scripts.countqa_eval --arm exemplar --variant matched \
  --features $VTB_WORK/features/countqa/<model>_<res>_full \
  --readouts $VTB_WORK/readouts/count-v1/<model> --seeds 0 1 2
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

- MAE and RMSE (the square root of the mean `sq_error` over every seed × image row)
- exact match, where a prediction counts as correct when `floor(pred + 0.5)` equals the answer
- per-bin values and sample counts (`counting.count_bin` of the answer)

Stage differences use the paired image bootstrap (`vtb.geometry_bootstrap.paired_metric_bootstrap`).
Images are identical across Stages, seed k is paired with seed k, and each image's metric is
averaged over seeds. It draws 10,000 resamples with bootstrap seed 0 and reports a 95%
percentile interval.
