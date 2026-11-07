import argparse
from pathlib import Path
import sys

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from projector import load_projector, merged_and_projected


MODEL_ID = "immanuelpeter/PixtralViT-Large-4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Mistral Large 4 visual features.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # trust_remote_code runs modeling_pixtral_large4.py, the parent's BF16 numeric order.
    tower = AutoModel.from_pretrained(args.model, dtype=dtype, trust_remote_code=True).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model)
    projector = load_projector(args.model).to(device=device, dtype=dtype).eval()
    inputs = processor(images=Image.open(args.image).convert("RGB"), return_tensors="pt")
    image_sizes = inputs["image_sizes"].tolist()

    with torch.inference_mode():
        pixels = inputs["pixel_values"].to(device, dtype)
        tower_tokens = tower(pixels, image_sizes=image_sizes).last_hidden_state.squeeze(0)
        merged, projected = merged_and_projected(projector, tower_tokens, image_sizes)

    print("image_sizes", image_sizes)
    print("tower", tuple(tower_tokens.shape))
    print("merged", tuple(merged.shape))
    print("projected", tuple(projected.shape))


if __name__ == "__main__":
    main()
