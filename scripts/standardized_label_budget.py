"""Run the archived channel-standardized label-budget diagnostic without changing primary probes."""
import json
from dataclasses import dataclass

import torch

from vtb import cache, probe, probe_run

PATCH_SOURCE = "r2:vtb/backups/v3/2026-10-19/proposed/reducer-channel-standardization.patch"
PATCH_SHA256 = "f5a2bc1dcda0e6be9a2d6abea9c1bcc9cb5817282ff6406489b5022dc65e330a"


@dataclass(frozen=True)
class StandardizedReducer:
    basis: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor

    def __call__(self, tokens):
        return ((tokens - self.mean) / self.scale) @ self.basis

    @classmethod
    def fit(cls, tokens, width, oversample=16):
        flat = tokens.reshape(-1, tokens.shape[-1]).float()
        mean = flat.mean(dim=0, keepdim=True)
        scale = flat.std(dim=0, keepdim=True).clamp_min(1e-6)
        _, _, v = torch.svd_lowrank((flat - mean) / scale, q=min(width + oversample, min(flat.shape)))
        return cls(v[:, :width].contiguous(), mean, scale)


def fit_reducer(features, split, width):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        return StandardizedReducer.fit(features[split.train], width)


def main():
    args = probe_run.build_parser().parse_args()
    if not args.match_capacity or args.readout != "attention":
        raise SystemExit("this diagnostic requires matched attention")
    labels_by_id = json.loads(args.labels.read_text())["labels"]
    cells = []
    for stage, layer in probe_run.selected_slices(args.run, args.depth_points, args.stages):
        if stage != "tower":
            raise ValueError("the label-budget diagnostic covers the final Tower Stage")
        tokens, ids, meta = cache.load(args.run, stage, layer)
        labels = torch.tensor([labels_by_id[i] for i in ids])
        full_split = probe_run.split_indices(len(ids))
        split = probe_run.subsample_train(full_split, labels, args.label_fraction)
        features = tokens.float()
        features = fit_reducer(features, split, args.width)(features)
        result = probe_run.run_cell(features, labels, split, len(set(labels_by_id.values())), args)
        result.update(model_id=meta["model_id"], stage=stage, layer_index=layer,
                      relative_depth=float(meta["relative_depth"]), token_width=tokens.shape[-1],
                      trainable_parameters=probe.parameter_count(probe.build(args.readout, features.shape[-1], len(set(labels_by_id.values())))),
                      label_fraction=args.label_fraction, train_images=len(split.train), full_train_images=len(full_split.train))
        cells.append(result)
    assert len(cells) == 1
    payload = dict(images=len(ids), label_fraction=args.label_fraction, train_images=len(split.train),
                   full_train_images=len(full_split.train), cells=cells,
                   diagnostic=dict(method="training-split sample standard deviation before seeded PCA",
                                   patch_source=PATCH_SOURCE, patch_sha256=PATCH_SHA256))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

