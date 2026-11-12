"""Independent official vLLM vision reference, shared by K3 and K3.1 Pro."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vtb.adapters.moonvit_v2 import CHECKPOINTS

pytestmark = pytest.mark.skipif(os.environ.get("VTB_SKIP_WEIGHTS") == "1", reason="VTB_SKIP_WEIGHTS=1")


@pytest.mark.parametrize("key", CHECKPOINTS)
@pytest.mark.parametrize("attention,dtype", [("official_sdpa", "float32"), ("sdpa_math", "bfloat16")])
def test_official_parent_forward(key, attention, dtype, tmp_path):
    root = os.environ.get("VTB_KIMI_VALIDATION_ROOT")
    if not root:
        pytest.skip("set VTB_KIMI_VALIDATION_ROOT to pinned source/config/export evidence")
    root = Path(root)
    out = tmp_path / "parity.json"
    checkpoint = CHECKPOINTS[key]
    config = root / f"{key}-parent-config.json"
    bundle = root / ("k3-bundle" if key == "moonvit_v2" else "bundle")
    subprocess.run([
        sys.executable, "-m", "scripts.validate_kimi_vllm",
        "--checkpoint", key,
        "--source", str(root / "reference/vllm/model_executor/models/kimi_k25_vit.py"),
        "--parent-config", str(config),
        "--bundle", str(bundle), "--out", str(out),
        "--device", os.environ.get("VTB_PARITY_DEVICE", "cpu"),
        "--dtype", dtype, "--attention", attention,
    ], check=True)
    report = json.loads(out.read_text())
    assert report["source_revision"] == checkpoint.revision
    assert report["passed"]
    assert report["attention_substitution"] == attention
    if dtype == "bfloat16":
        assert all(case["max_abs"] == 0 for case in report["cases"])
    assert len(report["cases"]) == 30
    assert {tuple(c["shape"][-2:]) for c in report["cases"]} == {
        (1024, 1024), (256, 4096), (256, 7168),
    }
