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
