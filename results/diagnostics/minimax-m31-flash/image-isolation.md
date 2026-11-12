# Image-isolation correction

The native HF MiniMax vision class packs its inputs into one attention sequence.
Unrelated benchmark images must not share that sequence. The original adapter
forwarded a packed external batch directly to it.

A CPU float32 test at 56 pixels held one black image fixed and added a white
image to its batch. All ten slices changed. The complete numerical evidence is
in batch-isolation-failure.json. Earlier single-image parity passed because it
did not exercise this dependency.

The shared adapter now calls the native Tower and Projector separately for each
image, then restores the external batch layout. Checkpoint weights, preprocessing,
patch packing, raster restoration and spatial merging are unchanged. External
batch size remains eight where prescribed. This is an inference isolation
correction, not a model architecture or reducer change.

The corrected adapter matches an independently loaded, SHA-pinned native parent
reference exactly in BF16 at 56 and 448 pixels. Cases include external batches
of one, two and eight, reversed order, and a changed neighboring image. All
200 batched slice comparisons in 20 cases are exact. The reference uses native
HF classes with independent source loading and coordinate-based preprocessing
and raster restoration. It is not a second modeling implementation or a full
language-model forward.

All features and dependent capability measurements produced before this
correction in this delivery are archived under
/tmp/vtb-delivery/issues/21/invalid-packed-attention and excluded from publication.
Raw weight identity, release reload, input hashes and weight-drift evidence remain
valid.

Fresh M3 primary and paired controls use the same corrected image isolation as
Flash. Historical published M3 files are preserved; their extraction provenance
does not establish this new validation scope. Additive figure-source overrides
select the fresh control files, including its standardized label-budget data.
Only those compatible measurements support the sibling checkpoint comparison. The training histories are not controlled, so the measurements do not attribute differences to particular training interventions.
