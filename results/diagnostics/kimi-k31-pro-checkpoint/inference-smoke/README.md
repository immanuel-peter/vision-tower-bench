# Published inference example gate

The current immutable Pro release has no examples/inference.py. The exact
published-example invocation failed with exit 2. The PR publication gate
remains blocked. Published model cards, repository structure and uploads
belong to other actors.

The proposed functional fix is examples/kimi_k31_pro_inference.py in this
bench. Its intended HF destination is examples/inference.py. The candidate
passed one CUDA BF16 image on physical GPU1 from a new isolated download of
the published immutable release. It produced finite Tower 1x1024x1024,
merged 1x256x4096 and projected 1x256x7168 outputs with two CPU threads.
This proves the bench-side candidate against fresh published files. It does
not prove execution of a published example, because that file is absent.

The source SHA, exact command, real image hash and original size, preprocessing,
immutable revision, device, runtime and exit are in candidate-cuda-smoke.json.
All downloaded snapshot hashes remained unchanged after inference. GPU1 was
verified empty before CUDA initialization and released afterward with zero
memory and no CUDA processes, including exclusive lock reacquisition.

hf-owner-functional-handoff.json identifies the exact required single-file
functional addition. No HF upload or card, manifest, license or layout edit
was performed. After owner publication, a new clean immutable snapshot and
an explicitly authorized CUDA smoke must pass before the PR opens.
Production official TORCH_SDPA BF16 parity remains failed; this functional
smoke makes no new conversion-parity claim. GPU0 evaluation was not changed.
