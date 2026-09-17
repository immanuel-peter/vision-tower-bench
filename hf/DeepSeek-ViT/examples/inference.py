import argparse
from pathlib import Path
import sys

from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision import Aligner, ViT


MODEL_ID = "immanuelpeter/DeepSeek-ViT"
PATCH_SIZE = 14


def bundle_file(model: str, name: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path / name
    return Path(hf_hub_download(model, name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DeepSeek-ViT visual features.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tower = ViT()
    tower.load_state_dict(load_file(bundle_file(args.model, "model.safetensors")))
    tower = tower.to(device=device, dtype=dtype).eval()
    aligner = Aligner()
    aligner.load_state_dict(load_file(bundle_file(args.model, "projector.safetensors")))
    aligner = aligner.to(device=device, dtype=dtype).eval()

    image = Image.open(args.image).convert("RGB")
    pixels = transforms.Compose([
        transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])(image)
    channels, height, width = pixels.shape
    rows, cols = height // PATCH_SIZE, width // PATCH_SIZE
    patches = pixels.reshape(channels, rows, PATCH_SIZE, cols, PATCH_SIZE)
    patches = patches.permute(1, 3, 0, 2, 4).reshape(rows * cols, channels, PATCH_SIZE, PATCH_SIZE)
    patches = patches.to(device=device, dtype=dtype)

    with torch.inference_mode():
        tokens = tower(patches, rows, cols)
        projected = aligner(tokens, rows, cols)

    print("tower", tokens.shape)
    print("projected", projected.shape)


if __name__ == "__main__":
    main()
