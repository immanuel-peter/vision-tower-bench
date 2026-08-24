# vision-tower-bench

A frozen-feature probing suite that measures what visual information survives each stage of open multimodal vision encoders, from raw tower layers through the learned projector into LLM embedding space. It publishes capability profiles, not a single leaderboard score.

## Language

**Tower**:
The self-attention vision encoder (ViT) inside a multimodal model, evaluated standalone with frozen weights.
_Avoid_: vision encoder, backbone, ViT (when referring to a specific model's encoder)

**Projector**:
The learned module that maps Tower output tokens into the language model's embedding space.
_Avoid_: adapter, MLP (when referring to this module generally)

**Connector**:
The full Tower-to-LLM pipeline: any spatial token merging plus the Projector.
_Avoid_: bridge

**Stage**:
The extraction point of features within a model. One of `tower` (raw Tower output), `merged` (after spatial merging, before the Projector's learned mapping — the Projector's actual input), or `projected` (after Projector). Stages are model-relative: a Stage a model lacks is reported N/A, not skipped.

**Relative Depth**:
A checkpoint position normalized as layer index over total Tower layers; the x-axis that makes Towers of different depths comparable.

**Capability Profile**:
A model's per-task, per-Stage result vector, published instead of a single aggregate intelligence score.
_Avoid_: leaderboard entry, overall score

**Transfer Probe**:
A single driving-scene experiment (e.g., KITTI depth) included as evidence of road-scene transfer; not part of the core AV-free evaluation identity.

**Perturbation Study**:
The controlled experiment where exactly one visual factor changes at a time across graded levels on otherwise identical real images, with transform parameters recorded per image.
_Avoid_: stress test, augmentation benchmark
