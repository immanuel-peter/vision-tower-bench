import argparse
from pathlib import Path
import runpy

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

BUNDLE = Path(__file__).absolute().parents[1]
load_projector = runpy.run_path(str(BUNDLE / "projector.py"))["load_projector"]


MODEL_ID = "immanuelpeter/Muse-Glimmer-Vision"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Muse Glimmer visual features.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tower = AutoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model)
    projector = load_projector(args.model)
    projector.to(device=device, dtype=dtype).eval()
    inputs = processor(images=Image.open(args.image).convert("RGB"), return_tensors="pt")

    with torch.inference_mode():
        merged = tower(
            pixel_values=inputs["pixel_values"].to(device=device, dtype=dtype),
            grid_thw=inputs["image_grid_thw"].to(device),
        ).last_hidden_state
        projected = projector(merged)

    print("merged", merged.shape)
    print("projected", projected.shape)


if __name__ == "__main__":
    main()
