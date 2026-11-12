"""Render the unchanged default figure CLI and record the files it actually reads."""
import hashlib
import json
import os
import runpy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

root = Path(__file__).parent
public = Path("results/diagnostics/kimi-k31-pro-checkpoint")
out = public / "final-default-figures"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "2"
assert json.loads((root / "full-experiment-inventory-hash-audit.json").read_text())["passed"]
assert json.loads((root / "final-regression-tests.json").read_text())["passed"]
inputs = {}
original = Path.read_text
results_root = Path("results").absolute()


def traced_read(path, *args, **kwargs):
    value = original(path, *args, **kwargs)
    if path.suffix == ".json" and path.absolute().is_relative_to(results_root):
        inputs[str(path.absolute().relative_to(Path.cwd()))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


previous_argv = sys.argv
try:
    Path.read_text = traced_read
    sys.argv = ["scripts/figures.py", "--out", str(out), "--format", "svg", "png", "pdf"]
    namespace = runpy.run_path("scripts/figures.py", run_name="__main__")
finally:
    Path.read_text = original
    sys.argv = previous_argv
assert len(namespace["TOWERS"]) == 17 and len(namespace["FIGURES"]) == 10
outputs = {}
for name in namespace["FIGURES"]:
    for extension in ("svg", "png", "pdf"):
        path = out / f"{name}.{extension}"
        assert path.stat().st_size > 100
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            ET.parse(path)
        elif extension == "png":
            with Image.open(path) as image:
                image.verify()
        else:
            assert path.read_bytes().startswith(b"%PDF")
        outputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
assert len(outputs) == 30
for path, expected in inputs.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
report = dict(passed=True, towers=list(namespace["TOWERS"]), figures=list(namespace["FIGURES"]),
              figure_files=30, formats=["svg", "png", "pdf"], cuda_visible_devices="", threads=2,
              actual_read_inputs_sha256=inputs, output_sha256=outputs,
              generator_sha256=hashlib.sha256(Path("scripts/figures.py").read_bytes()).hexdigest(),
              wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              cli="python scripts/figures.py --out " + str(out) + " --format svg png pdf")
(out / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"Validated30files from10defaultfigures across17Towers; traced{len(inputs)}actualJSONinputs")
