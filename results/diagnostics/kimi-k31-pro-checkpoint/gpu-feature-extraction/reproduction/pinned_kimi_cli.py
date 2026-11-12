"""Pin fresh K3 controls in adapter-loading benchmark CLIs without changing defaults."""

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

from vtb.adapters.moonvit_v2 import K3, K31_PRO, MoonViTV2Adapter
from vtb.extract import ADAPTERS

target, arguments = sys.argv[1], sys.argv[2:]
if target not in {"scripts.correspondence_run", "scripts.tower_throughput"}:
    raise SystemExit("wrapper is limited to the adapter-loading measurement CLIs")
records = []


def factory(checkpoint):
    class PinnedAdapter(MoonViTV2Adapter):
        model_id = checkpoint.model_id

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for filename, count in (("model.safetensors", 165), ("projector.safetensors", 3)):
                weights = load_file(hf_hub_download(self.model_id, filename, revision=checkpoint.bundle_revision))
                actual = self.model.state_dict() if filename == "model.safetensors" else self.projector.state_dict()
                if len(actual) != count or set(actual) != set(weights) or any(not torch.equal(actual[key].cpu(), value) for key, value in weights.items()):
                    raise ValueError("measurement adapter differs from immutable audited release")
            records.append(dict(checkpoint=dataclasses.asdict(checkpoint), identity_tensors=168,
                                exact_immutable_identity=True, device=str(self.device),
                                dtype=str(self.dtype), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                threads=torch.get_num_threads(), wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

    PinnedAdapter.checkpoint = checkpoint
    return PinnedAdapter


ADAPTERS["moonvit_v2"] = factory(dataclasses.replace(K3, bundle_revision="bcaedfccd1f07d23ab20de8a0a168668c27cacb3"))
ADAPTERS["kimi_k31_pro"] = factory(K31_PRO)
sys.argv = [target, *arguments]
runpy.run_module(target, run_name="__main__")
out = Path(arguments[arguments.index("--out") + 1])
out.with_suffix(".runtime.json").write_text(json.dumps(dict(target=target, adapters=records,
    backend_limitation="Production official TORCH_SDPA BF16 parity remains failed; mathematical-reference conversion accepted."), indent=2) + "\n")
