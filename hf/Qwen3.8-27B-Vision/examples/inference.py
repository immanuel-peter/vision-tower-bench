import argparse
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel


MODEL_ID = "immanuelpeter/Qwen3.8-27B-Vision"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Qwen3.8 visual features.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tower = AutoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model)
    inputs = processor(images=Image.open(args.image).convert("RGB"), return_tensors="pt")

    with torch.inference_mode():
        output = tower(
            hidden_states=inputs["pixel_values"].to(device=device, dtype=dtype),
            grid_thw=inputs["image_grid_thw"].to(device),
        )

    print("tower", output.last_hidden_state.shape)
    print("projected", output.pooler_output.shape)


if __name__ == "__main__":
    main()
