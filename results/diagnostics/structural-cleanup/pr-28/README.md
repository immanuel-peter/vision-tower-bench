# PR 28 cleanup validation

The common reducer and keyword-only API reproduce all four prior implementations
bit-for-bit. New recognition and perturbation producers serialize explicit method
identity. Actual-producer regressions reject mixed methods; unknown archived methods
require hash-bound provenance. Primary defaults and benchmark scores are unchanged.

- Regression suite: 164 passed, 52 skipped, 14 warnings in 18.93s
- Preservation: [preservation.json](preservation.json), including the original 64 standardized runs.
- Fresh published inference: [hf-release-smoke.json](hf-release-smoke.json), [stdout](inference.txt).
- Release: [immanuelpeter/PixtralViT-Large-4 at 9a8982a20c485af18d4c0270b11800a7fc0114c5](https://huggingface.co/immanuelpeter/PixtralViT-Large-4/tree/9a8982a20c485af18d4c0270b11800a7fc0114c5).
- Content hashes: [hf-files.json](hf-files.json). No weight upload.
- Combined checks: [integration audit](../integration-audit.json), [tests](../integration-tests.txt).

Command from the clean downloaded snapshot:
```bash
python examples/inference.py /tmp/vtb-delivery/data/imagenet100/validation/validation_000000.jpg
```
Physical GPU 0, logical cuda:0, BF16, batch 1. The fixed JPEG is 500x334 pixels,
SHA256 `dc4d456ce66cf18cc246831b3fe8b54133b507de981490a29f1623d86ddbeae3`; the example evaluates at 448x448.
Tower, merged and projected shapes are respectively `[1024, 2048]`, `[256, 2048]`
and `[256, 6144]`. Exit status is 0. The exclusive reservation was released.

The Mistral cross-precision/kernel sensitivity and Kimi official production BF16
disagreement retain their original failed outcomes. A successful inference smoke
does not change those scientific limitations. No experiment was retrained.
