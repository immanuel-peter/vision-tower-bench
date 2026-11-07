"""Extract independent-image MiniMax features from this downloaded bundle."""
import argparse
from pathlib import Path
import sys

from PIL import Image
import torch
from torchvision import transforms
from transformers import MiniMaxM3VLVisionModel

BUNDLE = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(BUNDLE))
from projector import load_projector


def preprocess(image: Image.Image) -> torch.Tensor:
    pixels = transforms.Compose([
        transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])(image)
    frames = pixels.unsqueeze(0).expand(2, -1, -1, -1)
    patches = frames.reshape(1, 2, 3, 16, 2, 14, 16, 2, 14)
    return patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(1024, 1176)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    device, dtype = torch.device(args.device), torch.bfloat16
    with Image.open(args.image) as original:
        original_size = original.size
        pixels = preprocess(original.convert("RGB")).to(device=device, dtype=dtype)
    model = MiniMaxM3VLVisionModel.from_pretrained(
        BUNDLE, dtype=dtype, local_files_only=True,
    ).to(device).eval()
    projector = load_projector(BUNDLE).to(device=device, dtype=dtype).eval()
    with torch.inference_mode():
        packed = model(pixels, torch.tensor([[1, 32, 32]], device=device)).last_hidden_state
        packed = packed.reshape(1, 1024, 1280)
        tower = packed.view(1, 16, 16, 2, 2, 1280).permute(0, 1, 3, 2, 4, 5).reshape(1, 1024, 1280)
        merged = tower.view(1, 16, 2, 16, 2, 1280).permute(0, 1, 3, 2, 4, 5).reshape(1, 256, 5120)
        projected = projector(packed.reshape(1024, 1280)).reshape(1, 256, 6144)
    print("bundle", BUNDLE)
    print("device", device, "dtype", dtype, "batch", 1, "image_size", original_size, "crop", (448, 448))
    for name, tokens, shape in (
        ("tower", tower, (1, 1024, 1280)),
        ("merged", merged, (1, 256, 5120)),
        ("projected", projected, (1, 256, 6144)),
    ):
        assert tuple(tokens.shape) == shape and tokens.dtype == torch.bfloat16
        assert torch.isfinite(tokens).all().item(), name
        print(name, tuple(tokens.shape), "dtype", tokens.dtype, "device", tokens.device)


if __name__ == "__main__":
    main()
