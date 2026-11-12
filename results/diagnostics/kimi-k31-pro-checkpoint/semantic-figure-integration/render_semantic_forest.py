from pathlib import Path
import hashlib,json
from scripts.figures import FIGURES,RESULTS,SOURCES,semantic_forest
out=Path("results/diagnostics/kimi-k31-pro-checkpoint/ready-default-figures")
name=next(key for key,value in FIGURES.items() if value is semantic_forest)
for suffix in ("svg","png","pdf"):semantic_forest(out/f"{name}.{suffix}")
paths=set(RESULTS.glob("bootstrap/semantic-*-vs-*.json"))|{RESULTS/path for path in SOURCES["semantic_checkpoint_comparisons"]}
(out/f"{name}.figure.json").write_text(json.dumps(dict(
 inputs={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)},
 explicit_checkpoint_sources=SOURCES["semantic_checkpoint_comparisons"],
 script_sha256=hashlib.sha256(Path("scripts/figures.py").read_bytes()).hexdigest(),
 benchmark_fits_changed=False,render_cpu_threads=2),indent=2)+"\n")
print(name,"source files",len(paths))
