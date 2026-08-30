import argparse
from pathlib import Path
import sys

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projector import load_projector


MODEL_ID = "immanuelpeter/MoonViT-K2.6"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Kimi K2.6 projected visual features.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tower = AutoModel.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=True)
    projector = load_projector(args.model)
    projector.to(device=device, dtype=dtype).eval()
    inputs = processor(images=Image.open(args.image).convert("RGB"), return_tensors="pt")

    with torch.inference_mode():
        merged = tower(
            inputs["pixel_values"].to(device=device, dtype=dtype),
            inputs["grid_thws"].to(device),
        )
        projected = projector(merged)

    print("merged", torch.stack(merged).flatten(2).shape)
    print("projected", torch.stack(projected).shape)


if __name__ == "__main__":
    main()
