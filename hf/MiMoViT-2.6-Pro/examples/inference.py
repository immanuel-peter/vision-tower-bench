import argparse
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel


MODEL_ID = "immanuelpeter/MiMoViT-2.6-Pro"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MiMo 2.6 Pro visual features.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=dtype).to(device).eval()
    inputs = processor(images=Image.open(args.image).convert("RGB"), return_tensors="pt")

    with torch.inference_mode():
        projected = model(inputs["pixel_values"].to(device, dtype), inputs["image_grid_thw"])

    print("grid_thw", inputs["image_grid_thw"].tolist())
    print("projected", tuple(projected.shape))


if __name__ == "__main__":
    main()
