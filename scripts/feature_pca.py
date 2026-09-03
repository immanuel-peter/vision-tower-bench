#!/usr/bin/env python3
"""Render per-Stage PCA maps for one image, so the Connector can be seen rather than tabulated.

Each Stage gets its own three-component PCA mapped to RGB. Bases are fitted per Stage
because the widths differ, so component signs are aligned against the `tower` map to keep
the panels comparable. Colour is still arbitrary; structure is the readable part.
"""

import argparse
from math import isqrt
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from vtb.extract import ADAPTERS


def component_maps(tokens: torch.Tensor) -> torch.Tensor:
    """Project one image's patch tokens onto three components, shaped (3, grid, grid)."""
    centred = tokens.float() - tokens.float().mean(0)
    _, _, basis = torch.pca_lowrank(centred, q=3)
    grid = isqrt(centred.shape[0])
    if grid * grid != centred.shape[0]:
        raise ValueError(f"{centred.shape[0]} tokens do not form a square grid")
    return (centred @ basis[:, :3]).reshape(grid, grid, 3).permute(2, 0, 1)


def align_signs(maps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Flip components that anti-correlate with the reference Stage at the same grid."""
    resampled = F.interpolate(reference.unsqueeze(0), size=maps.shape[-2:], mode="area")[0]
    for index in range(maps.shape[0]):
        a, b = maps[index].flatten(), resampled[index].flatten()
        if torch.dot(a - a.mean(), b - b.mean()) < 0:
            maps[index] = -maps[index]
    return maps


def to_image(maps: torch.Tensor, side: int) -> Image.Image:
    low = maps.amin(dim=(1, 2), keepdim=True)
    high = maps.amax(dim=(1, 2), keepdim=True)
    rgb = ((maps - low) / (high - low).clamp(min=1e-6) * 255).byte()
    return Image.fromarray(rgb.permute(1, 2, 0).numpy()).resize((side, side), Image.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="muse_glimmer", choices=sorted(ADAPTERS))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    adapter = ADAPTERS[args.model](resolution=args.resolution, device=args.device)
    picture = Image.open(args.image).convert("RGB")
    pixels = adapter.collate([adapter.preprocess()(picture)])

    # Keep the deepest layer of every Stage the adapter offers.
    final = {
        batch.stage: batch.tokens[0]
        for batch in adapter.extract(pixels, ["sample"])
        if batch.layer_index == adapter.num_layers
    }

    stem = f"{args.model}_{args.image.stem}"
    reference = component_maps(final["tower"])
    to_image(reference, args.resolution).save(args.out / f"pca_{stem}_tower.png")
    print(f"tower      {tuple(reference.shape[-2:])}")

    for stage in ("merged", "projected"):
        if stage not in final:
            continue
        maps = align_signs(component_maps(final[stage]), reference)
        to_image(maps, args.resolution).save(args.out / f"pca_{stem}_{stage}.png")
        print(f"{stage:<10} {tuple(maps.shape[-2:])}")

    # The source is model-independent, so it is named per image rather than per Tower.
    picture.resize((args.resolution, args.resolution)).save(
        args.out / f"pca_{args.image.stem}_source.png"
    )
    print(f"wrote {len(final) + 1} panels to {args.out}")


if __name__ == "__main__":
    main()
