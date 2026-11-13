"""Check the PixtralViT-Large-4 Tower and Projector against an independent Mistral forward.

The reference is mistral-inference 1.6.0, loaded from its SHA256-pinned wheel.
It reads the parent's native tensor names (``wq``, ``w1``, ``patch_merger``,
``vision_language_adapter``) and applies RoPE as complex multiplication on
interleaved channel pairs, so it never touches the HF conversion. The
reference's vision path is the four lines of
``Transformer.embed_vision_language_features``: ``vision_encoder`` ->
``pre_mm_projector_norm`` -> ``patch_merger`` -> ``vision_language_adapter``.

mistral-inference imports xformers for ``memory_efficient_attention``, which
is not installed. This script registers a substitute module. By default it
calls PyTorch's memory-efficient SDPA kernel, which PyTorch adapted from the
xformers kernel, in the input dtype and the reference's own (batch, sequence,
head, dim) layout. ``--reference-attention explicit`` instead computes
softmax(q k^T / sqrt(d)) v in float32, head by head. ``BlockDiagonalMask``
keeps the xformers semantics. ``tests/test_mistral_large4.py`` checks both against
direct formulas. Only one image is packed per reference call, and the
reference ``TransformerBlock`` does not forward the mask to attention anyway.
"""

from vtb.validate_mistral_large4 import main

if __name__ == "__main__":
    main()
