# MiMo 2.6 attention sink check, October 18 2026

Issue #5 asked whether the Hugging Face MiMo vision attention applies its sinks
the way serving engines do. MiMo 2.6 Pro and MiMo 2.6 Flash share this code. In
the 24 windowed blocks, `MiMoVisionAttention` (`vtb/models/mimo26.py`, extracted
unchanged from the pinned parents) adds a learned per-head sink bias to the
logits of key column 0, inside an inclusive ±64-token window. The four
full-attention blocks (0, 9, 18 and 27) have no sink. An LLM-style sink is a
separate softmax column that absorbs probability mass; it normalizes
differently and would change every output row.

## Method

The check pinned vLLM at `03ee2e134d6cc60c4cc84096c8492fcc971f0526`. Its MiMo
model file, `vllm/model_executor/models/mimo_v2_omni.py`, passes
`sinks_bias_key0=True` to vision attention. The check ran that revision's Triton
prefill kernel, `vllm/v1/attention/ops/triton_prefill_attention.py`, outside a
vLLM install. Only three imports were replaced: the platform object, the Triton
import and the `RCP_LN2` constant. The kernel and its launch function were
unchanged. SHA256s of both upstream files are in
[attention_audit.json](attention_audit.json).

`context_attention_fwd` ran with `is_causal=False`,
`sliding_window_q=sliding_window_k=64`, `sinks_bias_key0=True` and sinks spaced
evenly from -3 to 3 across the 32 heads. Inputs were two packed images of random
BF16 queries (32 heads) and keys and values (8 heads), head size 64, torch seed
26, on CUDA. The reference applies the HF semantics with PyTorch SDPA on each
image: KV heads repeated four times, the sink added to key column 0, and `-inf`
wherever |query - key| > 64.

## Results

| Tokens per image | Max absolute difference | RMS difference |
|---:|---:|---:|
| 36 | 0.015625 | 0.000651 |
| 196 | 0.01171875 | 0.000280 |
| 784 | 0.0078125 | 0.000211 |

Every case passed `rtol=atol=0.016`, within BF16 rounding. HF's key-0 sink and
inclusive ±64 window agree with vLLM's MiMo vision path. At 784 tokens, the
448 px grid, the window is active. SGLang's sink handling was not executed.

`test_sink_biases_key_zero_and_respects_window` in `tests/test_mimo26.py` checks
the same key-0 and window semantics on CPU against an explicit softmax. The
merger normalization is a separate cross-engine difference. vLLM uses RMSNorm
where HF and SGLang use LayerNorm. Swapping only that norm on the parity images
at 448 px moves projected outputs by up to 0.738 for Pro (RMS 0.0274, cosine
0.9746) and 0.101 for Flash (RMS 0.0135, cosine 0.9735). The releases follow HF
and SGLang; projected parity with vLLM is not established. The reference commits
are SGLang `b234b5c32e0d65e04a61912f85fc09e95e74a328` (`mimo_vl.py`) and vLLM
`03ee2e134d6cc60c4cc84096c8492fcc971f0526` (`mimo_v2_omni.py`).
