# HF card and inference follow-up

The card now follows the eight-heading house structure: 64 lines / 402 words before, 68 lines / 297 words after. The original audit text is retained in before-README.md. LICENSE-CODE is added to the Hub; .gitattributes is also retained in the bench bundle. No files were renamed and neither weight file was uploaded or changed.

The manifest covers all twelve distributed non-manifest files, including hidden attributes, examples and licenses. It records the two pinned source shards, 515 Tower tensors, eight Projector tensors, BF16 inference, original source dtype preservation and explicit self-hash exclusion. The example loads exclusively from its own snapshot root and forwards one image at a time.

The fresh clean download at b161cf1d04676db7a695762ac6208713fa352861 passed every distributed file hash. The card command ran verbatim with the image placeholder substituted:

    python examples/inference.py /tmp/vtb-delivery/data/imagenet100/validation/validation_000000.jpg

Working directory: the downloaded revision snapshot. Device: physical GPU 0, CUDA_VISIBLE_DEVICES=0, logical cuda:0. Dtype: torch.bfloat16. Batch: 1. CPU threads: 2. The real image is 500 by 334 pixels, SHA256 dc4d456ce66cf18cc246831b3fe8b54133b507de981490a29f1623d86ddbeae3; the example applies a 448-pixel bicubic center crop.

Outputs: Tower (1,1024,1280), merged (1,256,5120), projected (1,256,6144), all finite BF16 CUDA tensors. The driver was checked idle before Torch was imported. The example and outer flock processes exited, memory returned to 0 MiB, no compute processes remained, and gpu0.lock was reacquired and released. No inline Python snippet exists in this card; its sole shell block is the command above.

Full CPU regression: 152 passed, 51 skipped, including four new packaging tests. The original 103 measured payload checksums remain unchanged. Complete commands, image metadata, shapes, immutable tested revision and release proof are in report.json and cuda-smoke.txt.
