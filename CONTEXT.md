# vision-tower-bench

A frozen-feature probing suite that measures what visual information survives each stage
of an open multimodal vision encoder. It follows features from raw Tower layers through
the learned Projector into the LLM embedding space. Results are capability profiles, not
a single leaderboard score.

## Language

**Tower.**
The self-attention vision encoder (ViT) inside a multimodal model, evaluated standalone with frozen weights.
Avoid: vision encoder, backbone, ViT when referring to a specific model's encoder.

**Projector.**
The learned module that maps Tower output tokens into the language model's embedding space.
Avoid: adapter or MLP when referring to the whole module.

**Connector.**
The full Tower-to-LLM pipeline, including spatial token merging and the Projector.
Avoid: bridge.

**Stage.**
An extraction point within a model. `tower` is raw Tower output. `merged` is the
Projector input after spatial merging. `projected` is the Projector output. Stages are
model-relative, so report a Stage the model lacks as N/A.

**Relative Depth.**
The layer index divided by the total number of Tower layers. This is the x-axis for
comparing Towers of different depths.

**Capability Profile.**
A model's per-task, per-Stage result vector, published instead of a single aggregate intelligence score.
Avoid: leaderboard entry or overall score.

**Transfer Probe.**
A single driving-scene experiment, such as KITTI depth, that tests whether features
transfer to road scenes. It is not part of the core AV-free benchmark.

**Perturbation Study.**
A controlled experiment on real images. Each image family changes one visual factor at a
time across recorded levels.
Avoid: stress test or augmentation benchmark.

## Qwen checkpoint comparison

`qwen3_5` identifies Qwen3.8-27B. `qwen38_flash_next` identifies Qwen3.8
Flash-Next. Their Tower and merged widths are 1152 and 4608. Their Projector
outputs have widths 5120 and 2560, respectively. Tower and merged comparisons
change checkpoint weights. Projected comparisons also change output width.
Weight drift alone does not establish a training history or a reinitialization.

[Source and immutable release evidence](results/evidence/qwen38-flash-next/README.md)
records independent native forward parity and every tensor comparison. The
[isolated reducer diagnostic](results/diagnostics/qwen38-flash-next-conditioning/README.md)
recovers weak Flash-Next clean perturbation readouts with training-only channel
standardization before PCA. It establishes readout sensitivity. It does not
establish a checkpoint training cause or the same cause as MiMo's readout
failure. Primary Tower and merged perturbation measurements remain separate
and do not support intrinsic robustness claims given their low clean accuracy.
