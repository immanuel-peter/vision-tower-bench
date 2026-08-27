import argparse
from pathlib import Path

from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoImageProcessor, AutoModel


MODEL_ID = "immanuelpeter/MoonViT-V2"


def load_projector(model: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    model_path = Path(model)
    weights_path = (
        model_path / "projector.safetensors"
        if model_path.is_dir()
        else hf_hub_download(model, "projector.safetensors")
    )
    weights = load_file(weights_path)
    output_size, input_size = weights["proj.2.weight"].shape

    projector = nn.Module()
    projector.proj = nn.Sequential(
        nn.Linear(input_size, input_size, bias=False),
        nn.GELU(),
        nn.Linear(input_size, output_size, bias=False),
    )
    projector.post_norm = nn.RMSNorm(output_size, eps=1e-5)
    projector.load_state_dict(weights)
    return projector.to(device=device, dtype=dtype).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Kimi K3 projected visual features.")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="Hugging Face model ID or local model directory",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tower = AutoModel.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=True)
    projector = load_projector(args.model, device, dtype)

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    grid_thws = inputs["grid_thws"].to(device)

    with torch.inference_mode():
        merged_groups = tower(pixel_values, grid_thws)
        merged = torch.stack(merged_groups).flatten(2)
        projected = projector.post_norm(projector.proj(merged))

    print(projected.shape)


if __name__ == "__main__":
    main()
