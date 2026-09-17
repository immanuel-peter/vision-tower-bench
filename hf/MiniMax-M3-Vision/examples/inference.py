import argparse
from pathlib import Path
import sys

from PIL import Image
import torch
from transformers import MiniMaxM3VLImageProcessor, MiniMaxM3VLVisionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projector import load_projector


MODEL_ID = "immanuelpeter/MiniMax-M3-Vision"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MiniMax-M3 visual features.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tower = MiniMaxM3VLVisionModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = MiniMaxM3VLImageProcessor.from_pretrained(args.model)
    projector = load_projector(args.model)
    projector.to(device=device, dtype=dtype).eval()
    inputs = processor(images=Image.open(args.image).convert("RGB"), return_tensors="pt")

    with torch.inference_mode():
        merged = tower(
            pixel_values=inputs["pixel_values"].to(device=device, dtype=dtype),
            grid_thw=inputs["image_grid_thw"].to(device),
        ).last_hidden_state
        tokens = merged.reshape(-1, merged.shape[-1])
        projected = projector(tokens)

    print("tower", merged.shape)
    print("projected", projected.shape)


if __name__ == "__main__":
    main()
