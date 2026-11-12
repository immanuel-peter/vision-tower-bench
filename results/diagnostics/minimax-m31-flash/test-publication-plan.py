"""Preflight every final destination, including unfinished paired outputs, without copying."""
import ast,hashlib,json
from pathlib import Path
r=Path("/tmp/vtb-delivery/issues/21");source=r/"measurements";results=Path.cwd()/"results";diag=results/"diagnostics/minimax-m31-flash";control=diag/"m3-image-isolation"
script=r/"publish-measurements.py";tree=ast.parse(script.read_text());function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="destination")
namespace={"source":source,"results":results,"diag":diag,"control":control}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(script),"exec"),namespace)
paths=set(source.rglob("*.json"));inventory=json.loads((r/"expected-inventory.json").read_text())
paths.update(source/"bootstrap"/name for name in inventory["bootstrap"]["expected_files"])
assert len(paths)==103,len(paths)
planned={str(path):namespace["destination"](path) for path in paths}
assert len(set(planned.values()))==len(planned)
assert not any(target.exists() for target in planned.values()),"Existing published measurement would be replaced"
assert namespace["destination"](source/"throughput/minimax_m3.json")==control/"throughput/minimax_m3.json"
historical=results/"throughput/minimax_m3.json";assert historical not in planned.values()
report={"scope":"Destination preflight only; full content audits remain required before publication.","destinations":{path:str(target.relative_to(Path.cwd())) for path,target in sorted(planned.items())},"planned":103,"currently_existing_sources":sum(Path(path).is_file() for path in planned),"all_expected_paths_exist":all(Path(path).is_file() for path in planned),"historical_m3_throughput_sha256":hashlib.sha256(historical.read_bytes()).hexdigest() if historical.is_file() else None,"publisher_sha256":hashlib.sha256(script.read_bytes()).hexdigest()}
(r/"publication-plan-audit.json").write_text(json.dumps(report,indent=2)+"\n")
print("PASS:103unique additive destinations, no existing measurement replacement; M3throughput routed to control scope; actualsources",report["currently_existing_sources"])
