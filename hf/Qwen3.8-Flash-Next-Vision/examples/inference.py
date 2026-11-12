import argparse
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor, AutoModel


def main():
    parser = argparse.ArgumentParser(description="Extract all three Qwen visual Stages at 448 pixels.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default=str(Path(__file__).absolute().parents[1]))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model)
    crop = transforms.Compose([
        transforms.Resize(448, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(448),
    ])
    image = crop(Image.open(args.image).convert("RGB"))
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        output = model(
            hidden_states=inputs["pixel_values"].to(device=device, dtype=dtype),
            grid_thw=inputs["image_grid_thw"].to(device),
        )
    tower = output.last_hidden_state.reshape(1, -1, model.config.hidden_size)
    merged = tower.reshape(1, -1, model.config.hidden_size * model.config.spatial_merge_size**2)
    projected = output.pooler_output.reshape(1, merged.shape[1], -1)
    print("device", device, "dtype", dtype, "image", image.size)
    print("tower", tuple(tower.shape))
    print("merged", tuple(merged.shape))
    print("projected", tuple(projected.shape))


if __name__ == "__main__":
    main()
