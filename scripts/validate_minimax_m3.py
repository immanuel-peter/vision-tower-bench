"""Audit pinned MiniMax source identity, sibling drift and independent parent loading."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from safetensors.torch import load_file
from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as parent_code
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLConfig

from vtb.adapters import minimax_m3 as mm
from vtb.shards import load_prefixed


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = {"transformers": transformers.__version__, "source": {}, "identity": {}, "drift": {}, "forward": {}}
    source_path = Path(inspect.getfile(parent_code))
    report["reference"] = {"path": str(source_path), "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                           "scope": "Independent source-tensor loading into native HF parent classes; parent repo ships no modeling Python file."}
    states = {}
    for cp in (mm.M3, mm.FLASH):
        report["source"][cp.name] = {"repo": cp.repo, "revision": cp.revision}
        index = mm.source_json(cp, "model.safetensors.index.json")["weight_map"]
        raw = {}
        for prefix in (mm.TOWER_PREFIX, mm.PROJECTOR_PREFIX, mm.MERGE_MLP_PREFIX):
            shards = sorted({v for k, v in index.items() if k.startswith(prefix)})
            raw[prefix] = load_prefixed(cp.repo, shards, prefix, revision=cp.revision)
            report["source"][cp.name][prefix] = shards
        # Independent rename path, rather than the adapter's remap functions.
        tower = {k.replace("encoder.layers.", "layers.").replace("embeddings.patch_embedding.", "embeddings.proj."): v
                 for k, v in raw[mm.TOWER_PREFIX].items()}
        projector = dict(raw[mm.PROJECTOR_PREFIX])
        projector.update({k.replace("linear_", "merge_linear_"): v for k, v in raw[mm.MERGE_MLP_PREFIX].items()})
        states[cp.name] = (tower, projector)
        report["identity"][cp.name] = {}
        for filename, weights in zip(("model.safetensors", "projector.safetensors"), (tower, projector)):
            released = load_file(cp.bundle / filename)
            assert released.keys() == weights.keys()
            assert all(torch.equal(released[k], v) for k, v in weights.items())
            report["identity"][cp.name][filename] = {k: {"sha256": digest(v), "shape": list(v.shape), "dtype": str(v.dtype)} for k, v in weights.items()}

    for part, first, second in zip(("tower", "projector"), states[mm.M3.name], states[mm.FLASH.name]):
        rows = []
        for name in first:
            a, b = first[name].double().flatten(), second[name].double().flatten()
            rows.append({"name": name, "equal": torch.equal(a, b),
                         "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
                         "relative_l2": ((b-a).norm()/a.norm()).item()})
        assert any(not r["equal"] for r in rows)
        report["drift"][part] = rows

    for cp, cls in ((mm.M3, mm.MiniMaxM3Adapter), (mm.FLASH, mm.MiniMaxM31FlashAdapter)):
        config = MiniMaxM3VLConfig.from_pretrained(cp.repo, revision=cp.revision)
        reference = parent_code.MiniMaxM3VLVisionModel._from_config(config.vision_config, dtype=torch.bfloat16)
        reference.load_state_dict(states[cp.name][0], strict=True)
        reference = reference.to(args.device).eval()
        projector = parent_code.MiniMaxM3VLMultiModalProjector(config).to(dtype=torch.bfloat16)
        projector.load_state_dict(states[cp.name][1], strict=True)
        projector = projector.to(args.device).eval()
        adapter = cls(device=args.device)
        rows = []
        for resolution in (56, 448):
            rng = np.random.default_rng(21)
            pixels = rng.integers(0, 256, (resolution, resolution, 3), dtype=np.uint8)
            image = Image.fromarray(pixels)
            # Independent patch packing from normalized pixels in merge-block order.
            array = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float()/255
            array = (array-torch.tensor(mm.CLIP_MEAN)[:, None, None])/torch.tensor(mm.CLIP_STD)[:, None, None]
            side = resolution//14
            packed = []
            for y in range(0, side, 2):
                for x in range(0, side, 2):
                    for dy in range(2):
                        for dx in range(2):
                            patch = array[:, (y+dy)*14:(y+dy+1)*14, (x+dx)*14:(x+dx+1)*14]
                            packed.append(patch[:, None].expand(-1, 2, -1, -1).reshape(-1))
            values = torch.stack(packed)
            assert torch.equal(values, mm.Preprocess(resolution)(image))
            grid = torch.tensor([[1, side, side]], device=args.device)
            captures = {}
            handles = [reference.layers[i-1].register_forward_hook(
                lambda module, inp, out, i=i: captures.__setitem__(i, out.detach())) for i in adapter.depth_points()]
            with torch.inference_mode():
                output = reference(values.to(args.device, torch.bfloat16), grid).last_hidden_state
                expected_projected = projector(output.reshape(-1, 1280)).reshape(1, -1, 6144).cpu()
                nominal = reference.config.image_size
                reference.config.image_size = 672 if nominal != 672 else 2016
                altered = reference(values.to(args.device, torch.bfloat16), grid).last_hidden_state
                assert torch.equal(output, altered)
                reference.config.image_size = nominal
            for handle in handles:
                handle.remove()
            adapter.resolution = resolution
            actual = list(adapter.extract(mm.collate([values]), ["fixed-21"]))
            comparisons = []
            for batch in actual:
                if batch.stage == "projected":
                    expected = expected_projected
                else:
                    packed_tokens = captures[batch.layer_index].reshape(-1, 1280).cpu()
                    raster = torch.empty(1, side, side, 1280)
                    at = 0
                    for y in range(0, side, 2):
                        for x in range(0, side, 2):
                            for dy in range(2):
                                for dx in range(2):
                                    raster[0, y+dy, x+dx] = packed_tokens[at]
                                    at += 1
                    if batch.stage == "tower":
                        expected = raster.reshape(1, -1, 1280)
                    else:
                        expected = torch.stack([raster[:, y:y+2, x:x+2].reshape(1, 5120)
                                                for y in range(0, side, 2) for x in range(0, side, 2)], dim=1)
                delta = (batch.tokens.float()-expected.float()).abs().max().item()
                assert delta == 0, (cp.name, resolution, batch.stage, batch.layer_index, delta)
                comparisons.append({"stage": batch.stage, "layer": batch.layer_index, "max_abs": delta})
            rows.append({"resolution": resolution, "image_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                         "image_size_invariant": True, "comparisons": comparisons})
        report["forward"][cp.name] = rows
        del adapter, reference, projector
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
