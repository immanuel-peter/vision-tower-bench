import argparse
from pathlib import Path
import sys

from PIL import Image
import torch
from huggingface_hub import snapshot_download
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from vision import load_vision


MODEL_ID = "immanuelpeter/C-RADIOv4-H-3.5-Bolt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Nemotron Bolt visual features.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    bundle = Path(args.model) if Path(args.model).is_dir() else Path(snapshot_download(args.model))
    model = load_vision(bundle, dtype=dtype).to(device)

    image = Image.open(args.image).convert("RGB")
    pixels = transforms.Compose([
        transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
    ])(image).unsqueeze(0).to(device)

    for stage, tokens in model(pixels).items():
        print(stage, tuple(tokens.shape))


if __name__ == "__main__":
    main()
