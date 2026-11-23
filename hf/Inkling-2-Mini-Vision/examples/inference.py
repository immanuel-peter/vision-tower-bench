import argparse
from pathlib import Path
import runpy

from huggingface_hub import snapshot_download
from PIL import Image
import torch
from torchvision import transforms

BUNDLE = Path(__file__).absolute().parents[1]
vision = runpy.run_path(str(BUNDLE / "inkling2_vision.py"))

MODEL_ID = "immanuelpeter/Inkling-2-Mini-Vision"
RESOLUTION = 640


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Inkling 2 Mini visual features.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    bundle = Path(args.model) if Path(args.model).is_dir() else Path(snapshot_download(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tower, projector = vision["load_vision"](bundle, dtype)
    tower, projector = tower.to(device), projector.to(device)
    crop = transforms.Compose([
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
    ])
    pixels = vision["normalize"](crop(Image.open(args.image).convert("RGB")))[None].to(device, dtype)

    with torch.inference_mode():
        tokens = tower.forward_features(pixels)[:, 1:]
        merged = vision["merge"](tokens, RESOLUTION // vision["PATCH_SIZE"])
        projected = projector(merged)

    print("tower", tuple(tokens.shape))
    print("merged", tuple(merged.shape))
    print("projected", tuple(projected.shape))


if __name__ == "__main__":
    main()
