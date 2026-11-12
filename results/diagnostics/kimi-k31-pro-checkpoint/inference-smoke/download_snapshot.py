from pathlib import Path
import datetime,hashlib,json
from huggingface_hub import snapshot_download
root=Path(__file__).parent
inspection=json.loads((root/"published-source-inspection.json").read_text())
snapshot=Path(snapshot_download(inspection["repo"],revision=inspection["revision"],cache_dir=inspection["cache"],max_workers=2))
files={str(path.relative_to(snapshot)):hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshot.rglob("*") if path.is_file()}
assert set(files)==set(inspection["files"])
report=dict(repo=inspection["repo"],revision=inspection["revision"],snapshot=str(snapshot),cache=inspection["cache"],files_sha256=files,downloaded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),published_example_present=(snapshot/"examples/inference.py").is_file(),source="fresh immutable published snapshot; no export-dir/local-weights fallback")
(root/"fresh-download.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({key:value for key,value in report.items() if key!="files_sha256"}))
