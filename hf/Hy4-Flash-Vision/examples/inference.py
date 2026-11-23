"""Extract Hy4 flash Tower, merged and projected features from this downloaded bundle."""
import argparse
from pathlib import Path
import runpy

from PIL import Image
import torch
from torchvision import transforms

BUNDLE = Path(__file__).absolute().parents[1]
RESOLUTION, PATCH_SIZE = 512, 16


def preprocess(image: Image.Image) -> torch.Tensor:
    """Short-side bicubic resize and centre crop to 512 px, then mean and std 0.5."""
    return transforms.Compose([
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])(image.convert("RGB"))


def features(model, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
    """The last block (no final norm), the RMSNorm output grouped 2x2, and the Projector without newlines."""
    grid = (pixels.shape[-2] // PATCH_SIZE, pixels.shape[-1] // PATCH_SIZE)
    tower = model(pixels).last_hidden_state
    normed = model.perceive.norm(tower)
    return {
        "tower": tower,
        "merged": model.perceive.merge(normed, grid),
        "projected": model.perceive.project(normed, grid).flatten(2).transpose(1, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    modeling = runpy.run_path(str(BUNDLE / "modeling_hy4_flash_vision.py"))
    model = modeling["Hy4FlashVisionModel"].from_pretrained(BUNDLE, dtype=dtype, local_files_only=True)
    model = model.to(args.device).eval()
    with Image.open(args.image) as image:
        pixels = preprocess(image)[None].to(args.device, dtype)
    with torch.inference_mode():
        for name, tokens in features(model, pixels).items():
            print(name, tuple(tokens.shape))


if __name__ == "__main__":
    main()
