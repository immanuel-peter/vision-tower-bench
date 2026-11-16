"""Every Tower under its canonical results slug. Slugs are frozen even where they
name an older release than the weights they hold (`qwen3_5` is Qwen3.8,
`moonvit_v2` is Kimi K3).
"""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Tower:
    slug: str
    display: str
    model_id: str
    adapter_path: str
    colour: str

    def adapter(self) -> type:
        module, name = self.adapter_path.split(":")
        return getattr(import_module(module), name)


ROSTER: tuple[Tower, ...] = (
    Tower("dinov2", "DINOv2", "facebook/dinov2-large", "vtb.adapters.dinov2:DINOv2Adapter", "#1f77b4"),
    Tower("siglip2", "SigLIP2", "google/siglip2-so400m-patch14-384", "vtb.adapters.siglip2:SigLIP2Adapter", "#ff7f0e"),
    Tower("muse_glimmer", "Muse Glimmer", "immanuelpeter/Muse-Glimmer-Vision", "vtb.adapters.muse_glimmer:MuseGlimmerAdapter", "#2ca02c"),
    Tower("glm5", "GLM 5.3 Flash", "immanuelpeter/GLM-5.3-Flash-Vision", "vtb.adapters.glm5:GLM5Adapter", "#d62728"),
    Tower("nemotron_omni", "Nemotron 3 Omni", "immanuelpeter/C-RADIOv4-H-3-Omni", "vtb.adapters.nemotron_omni:NemotronOmniAdapter", "#9467bd"),
    Tower("kimi_k26", "Kimi K2.6", "immanuelpeter/MoonViT-K2.6", "vtb.adapters.kimi_k26:KimiK26Adapter", "#8c564b"),
    Tower("qwen3_5", "Qwen3.8", "immanuelpeter/Qwen3.8-27B-Vision", "vtb.adapters.qwen3_5:Qwen3_5Adapter", "#e377c2"),
    Tower("deepseek_v41", "DeepSeek V4.1", "immanuelpeter/DeepSeek-ViT", "vtb.adapters.deepseek_v41:DeepSeekV41Adapter", "#7f7f7f"),
    Tower("gemma4", "Gemma 4", "immanuelpeter/Gemma4-31B-Vision", "vtb.adapters.gemma4:Gemma4Adapter", "#bcbd22"),
    Tower("moonvit_v2", "Kimi K3", "immanuelpeter/MoonViT-V2", "vtb.adapters.moonvit_v2:MoonViTV2Adapter", "#17becf"),
    Tower("kimi_k31_pro", "Kimi K3.1 Pro", "immanuelpeter/MoonViT-K3.1-Pro", "vtb.adapters.moonvit_v2:KimiK31ProAdapter", "#31a354"),
    Tower("minimax_m3", "MiniMax M3", "immanuelpeter/MiniMax-M3-Vision", "vtb.adapters.minimax_m3:MiniMaxM3Adapter", "#393b79"),
    Tower("minimax_m31_flash", "MiniMax M3.1 Flash", "immanuelpeter/MiniMax-M3.1-Flash-Vision", "vtb.adapters.minimax_m3:MiniMaxM31FlashAdapter", "#9e9ac8"),
    Tower("nemotron35_super", "Nemotron 3.5 Super", "immanuelpeter/C-RADIOv4-H-3.5-Super", "vtb.adapters.nemotron35:Nemotron35SuperAdapter", "#637939"),
    Tower("nemotron35_bolt", "Nemotron 3.5 Bolt", "immanuelpeter/C-RADIOv4-H-3.5-Bolt", "vtb.adapters.nemotron35:Nemotron35BoltAdapter", "#b5cf6b"),
    Tower("dinov3", "DINOv3", "facebook/dinov3-vitl16-pretrain-lvd1689m", "vtb.adapters.dinov3:DINOv3Adapter", "#6baed6"),
    Tower("mimo26_pro", "MiMo V2.6 Pro", "immanuelpeter/MiMoViT-2.6-Pro", "vtb.adapters.mimo26:MiMo26ProAdapter", "#843c39"),
    Tower("mimo26_flash", "MiMo V2.6 Flash", "immanuelpeter/MiMoViT-2.6-Flash", "vtb.adapters.mimo26:MiMo26FlashAdapter", "#d6616b"),
    Tower("qwen38_flash_next", "Qwen3.8 Flash-Next", "immanuelpeter/Qwen3.8-Flash-Next-Vision", "vtb.adapters.qwen3_5:Qwen38FlashNextAdapter", "#a55194"),
    Tower("mistral_large4", "Mistral Large 4", "immanuelpeter/PixtralViT-Large-4", "vtb.adapters.mistral_large4:MistralLarge4Adapter", "#7b4173"),
)
SLUGS: tuple[str, ...] = tuple(tower.slug for tower in ROSTER)
BY_SLUG: dict[str, Tower] = {tower.slug: tower for tower in ROSTER}
