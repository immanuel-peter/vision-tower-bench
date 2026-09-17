import argparse
import json
from pathlib import Path
import sys

from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import transforms
from transformers import AutoConfig, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projector import load_projector


MODEL_ID = "immanuelpeter/C-RADIOv4-H"
PATCH_SIZE = 16


def bundle_file(model: str, name: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path / name
    return Path(hf_hub_download(model, name))


def pixel_shuffle(tokens: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
    batch, width, height, channels = tokens.size()
    tokens = tokens.view(batch, width, int(height * scale_factor), int(channels / scale_factor))
    tokens = tokens.permute(0, 2, 1, 3).contiguous()
    tokens = tokens.view(
        batch,
        int(height * scale_factor),
        int(width * scale_factor),
        int(channels / (scale_factor * scale_factor)),
    )
    return tokens.permute(0, 2, 1, 3).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract C-RADIOv4-H visual features.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    config = json.loads(bundle_file(args.model, "config.json").read_text())
    source = AutoConfig.from_pretrained(config["source"], trust_remote_code=True, revision=config["revision"])
    radio = AutoModel.from_config(source.vision_config, trust_remote_code=True)
    generator = radio.radio_model.model.patch_generator
    if not hasattr(generator, "video_embedder"):
        generator.video_embedder = torch.nn.Linear(
            2 * 3 * generator.patch_size * generator.patch_size,
            generator.embed_dim,
            bias=False,
        )
    radio.load_state_dict(load_file(bundle_file(args.model, "model.safetensors")), strict=False)
    radio = radio.to(device=device, dtype=dtype).eval()
    projector = load_projector(args.model).to(device=device, dtype=dtype).eval()

    image = Image.open(args.image).convert("RGB")
    pixels = transforms.Compose([
        transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
    ])(image).unsqueeze(0).to(device=device, dtype=dtype)

    with torch.inference_mode():
        features = radio(pixels).features
        height = pixels.shape[-2] // PATCH_SIZE
        width = pixels.shape[-1] // PATCH_SIZE
        merged = pixel_shuffle(features.reshape(1, height, width, -1)).reshape(1, -1, -1)
        projected = projector(merged)

    print("merged", merged.shape)
    print("projected", projected.shape)


if __name__ == "__main__":
    main()
