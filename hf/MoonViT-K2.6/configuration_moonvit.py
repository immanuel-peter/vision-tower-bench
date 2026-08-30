from transformers.configuration_utils import PretrainedConfig


class MoonViTConfig(PretrainedConfig):
    model_type = "moonvit_k26"

    def __init__(
        self,
        patch_size=14,
        init_pos_emb_height=64,
        init_pos_emb_width=64,
        init_pos_emb_time=4,
        pos_emb_type="divided_fixed",
        num_attention_heads=16,
        num_hidden_layers=27,
        hidden_size=1152,
        intermediate_size=4304,
        merge_kernel_size=(2, 2),
        video_attn_type="spatial_temporal",
        merge_type="sd2_tpool",
        _attn_implementation="eager",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.merge_kernel_size = merge_kernel_size
        self.video_attn_type = video_attn_type
        self.merge_type = merge_type
        self._attn_implementation = _attn_implementation
