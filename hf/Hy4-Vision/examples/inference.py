"""Extract Tower, merged, and projected features from a downloaded Hy4 bundle."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

BUNDLE = Path(__file__).absolute().parents[1]
RESOLUTION = 512


def preprocess(image: Image.Image) -> torch.Tensor:
    crop = transforms.Compose(
        [
            transforms.Resize(
                RESOLUTION, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return crop(image.convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = (
        AutoModel.from_pretrained(
            BUNDLE, trust_remote_code=True, local_files_only=True, dtype=dtype
        )
        .to(device)
        .eval()
    )
    with Image.open(args.image) as image:
        pixels = preprocess(image).unsqueeze(0).to(device=device, dtype=dtype)

    last = model.config.num_hidden_layers
    with torch.inference_mode():
        hidden, merged, projected = model(pixels, depths=(last,))

    print("tower", tuple(hidden[last].shape))
    print("merged", tuple(merged.shape))
    print("projected", tuple(projected.shape))


if __name__ == "__main__":
    main()
