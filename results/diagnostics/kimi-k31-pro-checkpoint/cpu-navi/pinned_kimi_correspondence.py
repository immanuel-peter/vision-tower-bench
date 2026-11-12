"""Run pinned correspondence with native feature integrity checks on every image."""

import dataclasses
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from scripts.extract_kimi_cpu import validate_batches
from vtb.adapters.moonvit_v2 import K3, K31_PRO, MoonViTV2Adapter
from vtb.extract import ADAPTERS

arguments = sys.argv[1:]
records = []


def factory(checkpoint):
    class PinnedAdapter(MoonViTV2Adapter):
        model_id = checkpoint.model_id

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.validated_images = 0
            for filename, count in (("model.safetensors", 165), ("projector.safetensors", 3)):
                weights = load_file(hf_hub_download(self.model_id, filename, revision=checkpoint.bundle_revision))
                actual = self.model.state_dict() if filename == "model.safetensors" else self.projector.state_dict()
                if len(actual) != count or set(actual) != set(weights) or any(not torch.equal(actual[name].cpu(), weight) for name, weight in weights.items()):
                    raise ValueError("loaded correspondence checkpoint differs from immutable release")
            records.append(self)

        def extract(self, inputs, image_ids):
            batches = list(super().extract(inputs, image_ids))
            validate_batches(batches, image_ids[0])
            self.validated_images += 1
            yield from batches

    PinnedAdapter.checkpoint = checkpoint
    return PinnedAdapter


ADAPTERS["kimi_k31_pro"] = factory(K31_PRO)
ADAPTERS["moonvit_v2"] = factory(dataclasses.replace(K3, bundle_revision="bcaedfccd1f07d23ab20de8a0a168668c27cacb3"))
sys.argv = ["scripts.correspondence_run", *arguments]
runpy.run_module("scripts.correspondence_run", run_name="__main__")
out = Path(arguments[arguments.index("--out") + 1])
payload = json.loads(out.read_text())
if len(records) != 1 or records[0].validated_images != 2 * payload["pairs"]:
    raise ValueError("correspondence image-integrity coverage differs from actual pairs")
runtime = dict(target="scripts.correspondence_run", adapters=[dict(
    checkpoint=dataclasses.asdict(adapter.checkpoint), identity_tensors=168, exact_immutable_identity=True,
    device=str(adapter.device), dtype=str(adapter.dtype), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    threads=torch.get_num_threads(), validated_image_forwards=adapter.validated_images,
    native_integrity=dict(all_ten_slices=True, finite_bf16=True, exact_shapes=True, raster_merge_equal=True),
    wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()) for adapter in records],
    backend_limitation="Production official TORCH_SDPA BF16 parity remains failed; independent mathematical-reference conversion accepted.")
out.with_suffix(".runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
