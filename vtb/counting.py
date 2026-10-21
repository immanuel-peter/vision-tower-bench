"""Counting geometry shared by FSC-147 preparation, the count readout and CountQA.

Coordinates are continuous source pixels. Pixel (row i, column j) covers
[j, j + 1) x [i, i + 1), so its centre is (j + 0.5, i + 0.5). A point at x lies
in a window [a, b) when a <= x < b. Every frame maps the source image into a
square canvas, and canvas positions are stored in unit coordinates [0, 1)^2.
Density targets, predictions and counts live on one declared TARGET_GRID x
TARGET_GRID grid over that canvas, whatever the model's patch grid.
"""

from dataclasses import dataclass
from math import ceil, erf, sqrt

import numpy as np
import torch
from torch.nn.functional import normalize

PROTOCOL = "count-v1"
TARGET_GRID = 64
# One target cell is 1/64 of the canvas side, 6 px for a 384 px FSC-147 crop.
SIGMA_CELLS = 1.0
TRUNCATE_SIGMAS = 3.0
# A clipped exemplar box stays valid when it keeps at least this area fraction.
MIN_BOX_AREA_KEPT = 0.5
# Letterbox padding colour, the ImageNet mean in 8-bit RGB.
LETTERBOX_FILL = (124, 116, 104)
BINS = ("le50", "50to200", "gt200")


def count_bin(count: float) -> str:
    """Bins use the cropped ground-truth count: <=50, (50, 200], >200."""
    if count <= 50:
        return BINS[0]
    return BINS[1] if count <= 200 else BINS[2]


@dataclass(frozen=True)
class SquareFrame:
    """A square canvas over a source image.

    ``left`` and ``top`` place the canvas origin in source pixels. They are
    negative for letterbox padding. ``side`` is the canvas side in source pixels.
    """

    left: int
    top: int
    side: int
    width: int
    height: int
    kind: str

    def points(self, xy: np.ndarray) -> np.ndarray:
        """Source pixel (x, y) rows to canvas unit coordinates."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return (xy - (self.left, self.top)) / self.side

    def boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Source (x0, y0, x1, y1) rows to canvas unit coordinates."""
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        return (boxes - (self.left, self.top, self.left, self.top)) / self.side

    @property
    def valid(self) -> tuple[float, float, float, float]:
        """The source image's area inside the canvas, in unit coordinates."""
        x0, y0, x1, y1 = self.boxes(np.array([0, 0, self.width, self.height]))[0]
        return (max(x0, 0.0), max(y0, 0.0), min(x1, 1.0), min(y1, 1.0))

    def as_dict(self) -> dict:
        return {"kind": self.kind, "left": self.left, "top": self.top, "side": self.side,
                "width": self.width, "height": self.height}


def centre_crop_frame(width: int, height: int) -> SquareFrame:
    """The FSC-147 training frame: the central min-side square, integer offsets."""
    side = min(width, height)
    return SquareFrame((width - side) // 2, (height - side) // 2, side, width, height, "centre_crop")


def letterbox_frame(width: int, height: int) -> SquareFrame:
    """The CountQA transfer frame: the whole image centred on a max-side square."""
    side = max(width, height)
    return SquareFrame(-((side - width) // 2), -((side - height) // 2), side, width, height, "letterbox")


def inside(unit_xy: np.ndarray) -> np.ndarray:
    """Points retained by the canvas, using the half-open boundary rule."""
    unit_xy = np.asarray(unit_xy, dtype=np.float64).reshape(-1, 2)
    return np.all((unit_xy >= 0.0) & (unit_xy < 1.0), axis=1)


def clip_box(box: np.ndarray) -> tuple[np.ndarray, float]:
    """Clip a unit box to the canvas. Returns the clipped box and its kept area fraction."""
    x0, y0, x1, y1 = np.asarray(box, dtype=np.float64)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"degenerate box {box}")
    clipped = np.array([max(x0, 0.0), max(y0, 0.0), min(x1, 1.0), min(y1, 1.0)])
    kept = max(clipped[2] - clipped[0], 0.0) * max(clipped[3] - clipped[1], 0.0)
    return clipped, kept / ((x1 - x0) * (y1 - y0))


def _cell_mass(centre: float, grid: int, sigma: float, radius: int) -> tuple[int, np.ndarray]:
    """Gaussian mass integrated over each cell near ``centre`` (grid units), clipped to the grid."""
    home = min(int(centre), grid - 1)
    first, last = max(home - radius, 0), min(home + radius, grid - 1)
    edges = np.arange(first, last + 2, dtype=np.float64)
    cdf = np.array([0.5 * (1.0 + erf((edge - centre) / (sigma * sqrt(2.0)))) for edge in edges])
    return first, np.diff(cdf)


def density_map(
    unit_xy: np.ndarray,
    grid: int = TARGET_GRID,
    sigma: float = SIGMA_CELLS,
    truncate: float = TRUNCATE_SIGMAS,
) -> np.ndarray:
    """Sum one Gaussian per retained point on a grid x grid canvas.

    Each kernel integrates the Gaussian over cell areas, is truncated at
    ``truncate`` sigma and at the canvas edge, then renormalized, so every point
    contributes exactly one. Points outside [0, 1)^2 raise.
    """
    unit_xy = np.asarray(unit_xy, dtype=np.float64).reshape(-1, 2)
    if not inside(unit_xy).all():
        raise ValueError("density_map takes retained points only")
    density = np.zeros((grid, grid), dtype=np.float64)
    radius = ceil(truncate * sigma)
    for x, y in unit_xy * grid:
        col, along_x = _cell_mass(x, grid, sigma, radius)
        row, along_y = _cell_mass(y, grid, sigma, radius)
        kernel = np.outer(along_y, along_x)
        density[row:row + len(along_y), col:col + len(along_x)] += kernel / kernel.sum()
    return density.astype(np.float32)


def coverage(rect: tuple[float, float, float, float], grid: int) -> np.ndarray:
    """Fraction of each cell's area inside a unit rectangle, shape (grid, grid)."""
    x0, y0, x1, y1 = rect
    edges = np.arange(grid + 1, dtype=np.float64) / grid

    def overlap(low: float, high: float) -> np.ndarray:
        return np.clip(np.minimum(edges[1:], high) - np.maximum(edges[:-1], low), 0.0, None) * grid

    return np.outer(overlap(y0, y1), overlap(x0, x1)).astype(np.float32)


def box_weights(boxes: np.ndarray, side: int) -> np.ndarray:
    """Area-overlap weights of each unit box on a side x side patch grid, each summing to one."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    weights = np.stack([coverage(tuple(box), side) for box in boxes]) if len(boxes) else np.zeros((0, side, side))
    mass = weights.sum(axis=(1, 2), keepdims=True)
    if (mass <= 0).any():
        raise ValueError("an exemplar box covers no patch area")
    return (weights / mass).astype(np.float32)


def prototype_similarity(
    features: torch.Tensor,
    boxes: list[np.ndarray],
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exemplar correlation channel for the conditioned arm, shape (images, 1, h, w).

    For each image, every box's features are averaged with area-overlap weights,
    and the box averages are averaged with equal weight into one prototype. The
    image's mean feature over valid patches is subtracted from both the prototype
    and every patch, and the channel is their cosine similarity. Any number of
    boxes from one upward is accepted, so a head trained with three FSC-147
    exemplars can read one CountQA exemplar.
    """
    images, dim, height, width = features.shape
    if height != width:
        raise ValueError("prototype similarity needs a square patch grid")
    if len(boxes) != images:
        raise ValueError(f"{len(boxes)} box sets for {images} images")
    out = torch.empty(images, 1, height, width, dtype=features.dtype)
    for index, image_boxes in enumerate(boxes):
        if len(image_boxes) == 0:
            raise ValueError(f"image {index} has no exemplar box")
        grid = features[index].float()
        weights = torch.from_numpy(box_weights(image_boxes, height)).to(grid.device)
        prototype = torch.einsum("khw,dhw->kd", weights, grid).mean(dim=0)
        mask = (valid[index].float().to(grid.device) if valid is not None
                else torch.ones(height, width, device=grid.device))
        centre = (grid * mask).sum(dim=(1, 2)) / mask.sum().clamp(min=1e-6)
        patches = normalize((grid - centre[:, None, None]).flatten(1), dim=0)
        query = normalize(prototype - centre, dim=0)
        out[index, 0] = (query @ patches).view(height, width).to(features.dtype)
    return out


def condition(features: torch.Tensor, boxes: list[np.ndarray], valid: torch.Tensor | None = None) -> torch.Tensor:
    """Append the exemplar correlation channel to (images, dim, h, w) features."""
    return torch.cat([features, prototype_similarity(features, boxes, valid)], dim=1)


def masked_count(density: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Sum a (images, 1, G, G) density on the target grid, weighting cells by valid area."""
    density = density.squeeze(1)
    if valid is not None:
        density = density * valid.to(density)
    return density.flatten(1).sum(dim=1)
