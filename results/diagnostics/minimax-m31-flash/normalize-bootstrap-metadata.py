import json
from pathlib import Path
root=Path("/tmp/vtb-delivery/issues/21")
for path in (root/"measurements/bootstrap").glob("*.json"):
 d=json.loads(path.read_text())
 protocol=d.get("protocol",{})
 for key in ("first_source","second_source"):
  if key in protocol:
   value=Path(protocol[key])
   if not value.is_absolute():
    assert value.parts[0]=="results",value
    continue
   relative=value.relative_to(root/"measurements")
   prefix=Path("results/diagnostics/minimax-m31-flash/m3-image-isolation") if value.name.startswith("minimax_m3_") else Path("results")
   protocol[key]=str(prefix/relative)
 if path.name.startswith("semantic-m31-vs-m3-"):
  assert d["first"]["stage"]==d["second"]["stage"]
  protocol["stage"]=d["first"]["stage"]
 path.write_text(json.dumps(d,indent=2)+"\n")
