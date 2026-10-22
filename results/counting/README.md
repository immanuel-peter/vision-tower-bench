# Counting (protocol count-v4)

Class-agnostic counting with a dense density readout on frozen final-Stage patch
tokens (issue #8), plus a frozen transfer column on a curated CountQA subset
(issue #10, [countqa/](countqa/)). The geometry runner trains the readout:
`vtb.geometry_run --task count`.

## Data

FSC-147 (Ranjan et al., *Learning To Count Everything*, CVPR 2021). Annotations,
split and class list come from `cvlab-stonybrook/LearningToCountEverything` at
commit `00ea1888c3a7c495ae06db0bddc3b90b7db8d52f`. Their sha256 values match the
copies in the `isentropic/FSC147` mirror at `3e420cb6537e803dd6d4516623ce82a79c0317b8`.
The official Google Drive image archive refused downloads (quota exceeded), so the
images come from that mirror. All 6,135 image sizes match the official
annotation's `W * ratio_w` and `H * ratio_h` within 1 px and the shapes of the
authors' density maps (archive.org). Image bytes have not been compared with the
Drive archive. `scripts/fetch_fsc147.sh` and `scripts/prep_fsc147.py
--reference-images` close that check when Drive serves the archive.

The official class-disjoint split is used: 3,659 train, 1,286 val, 1,190 test
images, with 89, 29 and 29 classes. The images have no stated licence, so no FSC-147
pixels are published here; FSC-147 figures show density maps only.

## Targets

Coordinates are continuous source pixels: pixel (row i, column j) covers
[j, j+1) x [i, i+1). A point lies in a window [a, b) when a <= x < b.

1. Crop. Each image is cropped to its central min-side square at integer
   offsets and written as a PNG at native resolution. Every release image is
   384 px tall, so each crop is 384 x 384. The adapter's bicubic short-side
   resize then becomes one square resize: an upsampling of 1.1667x to 448 px,
   or 1.3333x to 512 px for DINOv3. DIODE images, by contrast, are downsampled.
2. Recount. A dot is kept if it falls inside the crop under the half-open rule.
   45 release dots in 33 images sit exactly on the image's own closing edge (x == width or
   y == height). They mark objects in the image, so dots within 1 px outside the image
   are first clamped onto it. The crop keeps 280,062 of 343,693 release dots. Count bins
   use the retained count: <=50, 50 < c <= 200, > 200.
3. Density. Targets live on one declared 64 x 64 grid over the crop, whatever a
   model's patch grid. Each dot adds a Gaussian with sigma = 1 grid cell (6 source px).
   The Gaussian is integrated over cell areas, truncated at 3 sigma and at the crop
   edge, then renormalized. Every density therefore sums to the retained count
   exactly, edges included. Targets are never resampled. A count is the sum of a
   prediction on that grid.
4. Exemplars. Release boxes 0 to 2 are mapped through the crop and clipped. A
   box stays valid if it keeps at least half its area. 5,301 images keep all three
   boxes: 3,214 train, 1,068 val and 1,019 test. Only these are eligible for the
   exemplar arm. No box is ever replaced.

## Readout

`CountHead` (`vtb/geometry.py`) is the geometry `MultiscaleHead`, with hidden width
512 and 1x1 convolutions, followed by softplus on every head pixel. Exact
area-overlap weights then average head pixels onto the 64 x 64 grid. The head
predicts density times 100, and the loss is pixel MSE on that scaled density.
Gemma 4 pads its image to 480 px, and DeepSeek V4.1's merged and projected grids
span 462 px. Their grids therefore extend past the image, and the head maps only
the image part onto the target grid.

The protocol id changed three times during the pilot, before any matrix result
was kept:

- count-v1 used a final ReLU. Its Qwen3.8 Tower cell collapsed to an all-zero map
  (validation MAE 51.437, the score of predicting zero everywhere). This happened
  at every rate from 3e-3 up, and for two of three seeds at the selected 1e-3.
- count-v2 replaced the ReLU with softplus. Rates of 1e-2 and above still
  exploded in the first epoch and then sat at the zero map. One cell selected the
  grid floor, 1e-4.
- count-v3 moved the six rates down to 1e-5 to 3e-3. Its first cell selected the
  new top edge on a flat validation surface.
- count-v4 extends the search past an edge selection, as described below.

Training follows the geometry protocol: batch 8, 10 epochs, AdamW with weight
decay 1e-4, cosine schedule. The rate grid is six points one decade below the
geometry grid, {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}. Rates are searched with seed 0
and selected on official-validation count MAE. When the selection lands on an
edge of the searched rates, the search adds the next half-decade rate outward. It
does this at most twice, and stops when the new rate collapses or does not
improve. A fit collapses if it predicts less than a tenth of the validation mass,
or the same count for every image. Collapsed search fits are recorded and skipped.
If any final seed collapses at the selected rate, that rate is treated as unstable and every seed retrains at the next-best validation rate, at most twice. The cell records the original selection and each fallback, and a cell that collapses at every candidate fails rather than averaging a collapse. This rule was added after 13 production cells had finished. None of them had a collapsed seed, so the rule leaves them unchanged; DeepSeek V4.1's raw exemplar Tower cell was the first to trigger it. The pilot
also trained two cells for 20 epochs at their selected rates. Qwen3.8 merged at
1e-4 improved from 20.8 to 19.0 test MAE (one seed), and Kimi K2.6 projected at
1e-3 went from 14.8 to 15.1. Low-rate readouts are therefore not fully converged,
and Stage comparisons hold at a fixed 10-epoch budget.
The selected rate is then refit with seeds 0, 1 and 2. Every fit records
per-epoch training loss and validation MAE. Matched capacity reduces features to
512 dimensions with deterministic PCA fit on the official training split only.
That PCA is shared by both arms and saved.

Two arms:

- Exemplar-free (`free`): every final Stage the model has (Tower; merged and
  projected for the 13 multimodal Towers). 42 Stages x {raw, matched} = 84 cells.
- Exemplar-conditioned (`exemplar`): final Tower only, raw and matched, so 32 cells.
  Each box's patch features are averaged with area-overlap weights. The box
  averages are then averaged into one prototype. The image's mean patch feature is
  subtracted from the prototype and from every patch, and their cosine similarity
  becomes one extra input channel. That channel adds 512 head parameters. The same
  saved head accepts any number of boxes, so it can read one CountQA exemplar.
  It trains on eligible images only. Arm comparisons use the 1,019 common test
  images. Conditioning is not extended to merged or projected Stages.

Each cell saves the head of every final seed, the PCA, and the metadata needed
to reload them through `geometry_run.load_readout`. Per-image test predictions
are kept with signed, absolute and squared errors and the bin. Validation
predictions are kept in the run workspace. The conditioned arm also scores
official validation with release box 0 alone. This is a transfer diagnostic and is
never used for selection.

## Reproduce

```bash
source $VTB_WORK/env.sh            # bundle paths; no credentials
scripts/fetch_fsc147.sh $VTB_WORK/data/fsc147/raw    # FALLBACK=hf if Drive refuses
uv run python scripts/prep_fsc147.py --raw ... --out $VTB_WORK/data/fsc147
GPUS=0,1 ROSTER=$VTB_WORK/manifests/roster.json \
  scripts/count_matrix.sh $VTB_WORK/features/fsc147 $VTB_WORK/data/fsc147 \
  $VTB_WORK/results/fsc147 $VTB_WORK/readouts/count-v4 <model...>
```

Kimi K2.6 and MiniMax M3 must extract at batch 1, because larger batches mix
features across images. The roster manifest sets this.
