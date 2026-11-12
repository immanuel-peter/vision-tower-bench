"""Fetch only pinned official Kimi reference source, configs and model cards."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

from vtb.adapters.moonvit_v2 import CHECKPOINTS

VLLM_REVISION = "1ecc9cc4ce6626352df39e4364579ec376c10761"
FILES = (
    "vllm/model_executor/models/registry.py",
    "vllm/models/kimi_k3/__init__.py",
    "vllm/models/kimi_k3/nvidia/model.py",
    "vllm/model_executor/models/kimi_k25_vit.py",
    "vllm/model_executor/layers/attention/mm_encoder_attention.py",
    "vllm/model_executor/layers/rotary_embedding/common.py",
    "vllm/model_executor/layers/rotary_embedding/packed_qk_rope.py",
    "vllm/v1/attention/ops/vit_attn_wrappers.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for filename in FILES:
        path = args.out / "reference" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/vllm-project/vllm/{VLLM_REVISION}/{filename}"
        path.write_bytes(urllib.request.urlopen(url).read())
    for key, checkpoint in CHECKPOINTS.items():
        for name, target in (("config.json", f"{key}-parent-config.json"),
                             ("README.md", f"{key}-parent-README.md")):
            source = hf_hub_download(checkpoint.repo, name, revision=checkpoint.revision)
            (args.out / target).write_bytes(Path(source).read_bytes())
    registry = (args.out / "reference" / FILES[0]).read_text()
    if '"KimiK31ForConditionalGeneration"' not in registry:
        raise ValueError("pinned official registry does not register Pro")
    report = {"repo": "vllm-project/vllm", "revision": VLLM_REVISION,
              "files": {name: hashlib.sha256((args.out / "reference" / name).read_bytes()).hexdigest()
                        for name in FILES}}
    (args.out / "reference" / "source-manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
