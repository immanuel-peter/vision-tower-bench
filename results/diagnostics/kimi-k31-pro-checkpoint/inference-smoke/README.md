# Published inference example gate

The published-example gate now passes at immutable HF revision
5311e0408a6107f78f8c70bb70f7aa205b94e724. The parent performed the authorized
single-file publication, adding examples/inference.py unchanged from the
bench-side candidate. Every previous published file hash is unchanged.
The worker independently confirmed the immutable Hub revision and rehashed
the fresh snapshot. No further HF edits or uploads are authorized here.

The exact command was python examples/inference.py IMAGE from that clean
published snapshot. It exited zero in 7.606 seconds on physical GPU1,
logical cuda:0, CUDA BF16, batch one and two CPU threads. The actual outputs
were Tower 1x1024x1024, merged 1x256x4096 and projected 1x256x7168. The fixed
real image was 500x334 before bicubic resize and center crop to 448.
Exact command, image hash, preprocessing, device, runtime, revision and output
shapes are recorded under published-example-pass. GPU1 was verified empty
before initialization and released afterward with zero memory and no processes.

The original missing-file exit2 and separate bench-candidate CUDA proof remain
preserved as historical evidence. resolution.json records the superseding pass.
The benchmark adapter remains pinned to original conversion release 6ab95e2;
the new revision adds only the functional example. No benchmark fit, cache or
methodology changed. Production official TORCH_SDPA BF16 parity still fails;
the functional smoke does not reclassify that numerical limitation.
