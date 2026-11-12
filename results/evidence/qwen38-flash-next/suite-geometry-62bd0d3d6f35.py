"""Issue 22 resumable suite. Execution requires a fresh explicit GPU grant."""
from contextlib import nullcontext
import argparse, datetime, fcntl, hashlib, json, math, os, shlex, subprocess, time
from pathlib import Path

ROOT=Path("/home/shadeform/.t3/worktrees/vision-tower-bench/t3-b75c6e2e")
E=Path("/tmp/vtb-delivery/issues/22")
D=Path("/tmp/vtb-delivery/data")
MODEL="qwen38_flash_next"
DEPTHS=[3,7,10,14,17,20,24,27]
SLICES=[("tower",d) for d in DEPTHS]+[("merged",27),("projected",27)]
STAGES=["tower","merged","projected"]
MID={"qwen38_flash_next":"immanuelpeter/Qwen3.8-Flash-Next-Vision","qwen3_5":"immanuelpeter/Qwen3.8-27B-Vision"}
P=".venv/bin/python"
CORR_ROOT={name:D/"correspondence"/name/leaf for name,leaf in [("navi","navi_v1"),("scannet","scannet_test_1500"),("spair","SPair-71k")]}
jobs=[]

def add(name,argv,output,kind="json",needs=(),**expect):
    jobs.append(dict(name=name,argv=[str(x) for x in argv],output=str(output),kind=kind,needs=[str(x) for x in needs],expect=expect))

def runpath(task,model,pool):
    return E/"features"/task/f"{model}_448_{pool}"

def output(task,name):
    return E/"measurements"/task/(name+".json")

def build():
    for model in [MODEL,"qwen3_5"]:
        for task,images,pool in [("geometry",D/"diode/val",None),("recognition",D/"imagenet100/validation",4),("kitti",D/"kitti/val",None)]:
            out=runpath(task,model,"pool4" if pool else "full")
            argv=[P,"-m","vtb.extract","--model",model,"--images",images,"--out",out.parent,"--resolution","448","--device","cuda:0","--batch-size","8","--workers","2"]
            if pool: argv+=["--pool","4"]
            if task=="kitti": argv+=["--final-stages","tower","projected"]
            elif model=="qwen3_5": argv+=["--final-stages",*STAGES]
            slices=SLICES if model==MODEL and task!="kitti" else [(s,27) for s in (["tower","projected"] if task=="kitti" else STAGES)]
            add(f"extract-{task}-{model}",argv,out,kind="cache",needs=[images,D/("diode.READY" if task=="geometry" else "kitti.READY" if task=="kitti" else "imagenet100.READY")],model=model,slices=slices,pool=pool,images=str(images))
    # Priority: correspondence and projected/Tower geometry intervals.
    for model in [MODEL,"qwen3_5"]:
        for dataset in ["spair","scannet","navi"]:
            out=output("correspondence",f"{model}_correspondence_{dataset}")
            add(f"correspondence-{model}-{dataset}",[P,"-m","scripts.correspondence_run","--model",model,"--dataset",dataset,"--root",CORR_ROOT[dataset],"--out",out,"--device","cuda:0","--resolution","448"],out,kind="correspondence",needs=[D/"correspondence.READY"],model=model,dataset=dataset,cells=3)
    for task in ["depth","normal"]:
        for arm in ["matched","raw"]:
            run=runpath("geometry",MODEL,"full");out=output("bootstrap",f"geometry-{task}-{arm}-{MODEL}-projected-vs-tower")
            argv=[P,"-m","scripts.geometry_bootstrap","--first-run",run,"--first-name","projected","--first-stage","projected","--first-layer","27","--second-run",run,"--second-name","tower","--second-stage","tower","--second-layer","27","--targets",D/"diode/val_targets.npz","--task",task,"--arm",arm,"--device","cuda:0","--out",out]
            add(f"bootstrap-projector-{task}-{arm}",argv,out,needs=[run,D/"diode.READY"])
    for task in ["recognition","geometry","kitti"]:
        run=runpath(task,MODEL,"pool4" if task=="recognition" else "full")
        for readout in (["attention","mean"] if task=="recognition" else ["depth","normal"] if task=="geometry" else ["depth"]):
            for arm in (["raw","matched"] if task!="kitti" else ["matched"]):
                for stage,layer in (SLICES if task!="kitti" else [("tower",27),("projected",27)]):
                    name=f"{MODEL}_{readout}_{arm}_{stage}_L{layer:02d}"
                    out=output(task+"/cells",name)
                    argv=[P,"-m","vtb.probe_run" if task=="recognition" else "vtb.geometry_run","--run",run,"--device","cuda:0","--seeds","3","--out",out]
                    argv+=["--labels",D/"imagenet100/validation_labels.json","--readout",readout,"--stages",stage,"--depth-points",str(layer)] if task=="recognition" else ["--targets",D/("kitti" if task=="kitti" else "diode")/"val_targets.npz","--task",readout,"--only",f"{stage}:{layer}"]
                    if arm=="matched": argv+=["--match-capacity"]
                    add("kitti-"+name if task=="kitti" else name,argv,out,kind="matrix",needs=[run],model=MODEL,cells=1,slices=[(stage,layer)],grid=11 if task=="recognition" else 6,images=13000 if task=="recognition" else 771 if task=="geometry" else 1000)
    run=runpath("recognition",MODEL,"pool4")
    for readout in ["attention","mean"]:
        for fraction,tag in [("0.01","f001"),("0.05","f005"),("0.20","f020"),("1.0","f10")]:
            out=output("label-budget",f"{MODEL}_probe_{readout}_matched_{tag}")
            add(f"labels-{readout}-{tag}",[P,"-m","vtb.probe_run","--run",run,"--labels",D/"imagenet100/validation_labels.json","--device","cuda:0","--readout",readout,"--match-capacity","--stages","tower","--depth-points","27","--label-fraction",fraction,"--out",out],out,kind="matrix",needs=[run],model=MODEL,cells=1,slices=[("tower",27)],grid=11,images=13000)
    out=output("label-budget",MODEL+"_throughput")
    add("throughput",[P,"-m","scripts.tower_throughput","--model",MODEL,"--images",D/"imagenet100/validation","--resolution","448","--batch-size","8","--workers","2","--device","cuda:0","--out",out],out,kind="throughput",needs=[D/"imagenet100.READY"])
    for model in [MODEL,"qwen3_5"]:
        cache=E/"features/perturbation"
        out=output("perturbation",model+"_perturbation")
        add("perturbation-extract-"+model,[P,"-m","scripts.perturbation_extract","--model",model,"--identity-images",D/"imagenet100/validation","--condition-images",D/"perturbation","--transform-manifest",D/"perturbation/transforms.manifest.json","--out",cache,"--batch-size","8","--resolution","448","--workers","2","--device","cuda:0"],cache/"identity"/f"{model}_448_pool4",kind="perturbation-cache",needs=[D/"perturbation.READY"],model=model,slices=[(s,27) for s in STAGES],pool=4,images=str(D/"imagenet100/validation"))
        add("perturbation-score-"+model,[P,"-m","scripts.perturbation_run","--clean-run",cache/"identity"/f"{model}_448_pool4","--features-root",cache,"--model-dir",f"{model}_448_pool4","--labels",D/"imagenet100/validation_labels.json","--transform-manifest",D/"perturbation/transforms.manifest.json","--out",out,"--device","cuda:0"],out,kind="perturbation",needs=[cache/"identity"/f"{model}_448_pool4"],model=model,cells=3)
    for dataset in ["navi","scannet","spair"]:
        source=output("correspondence",f"{MODEL}_correspondence_{dataset}")
        out=output("bootstrap",f"correspondence-{dataset}-{MODEL}-projected-vs-tower")
        add("bootstrap-projector-"+dataset,[P,"-m","scripts.correspondence_bootstrap","--run",source,"--out",out],out,needs=[source])
        for stage in STAGES:
            second=output("correspondence",f"qwen3_5_correspondence_{dataset}")
            out=output("bootstrap",f"correspondence-{dataset}-flash-next-vs-27b-{stage}")
            add(f"bootstrap-sibling-{dataset}-{stage}",[P,"-m","scripts.cross_run_bootstrap","correspondence","--first",source,"--first-name",MODEL,"--second",second,"--second-name","qwen3_5","--stage",stage,"--out",out],out,needs=[source,second])
    for fraction,tag in [("0.01","f001"),("0.05","f005"),("0.20","f020"),("1.0","f10")]:
        run=runpath("recognition",MODEL,"pool4")
        out=output("diagnostics/standardized-label-budget",f"{MODEL}_probe_attention_matched_{tag}")
        add("standardized-labels-"+tag,[P,"-m","scripts.standardized_label_budget","--run",run,"--labels",D/"imagenet100/validation_labels.json","--device","cuda:0","--readout","attention","--match-capacity","--stages","tower","--depth-points","27","--label-fraction",fraction,"--out",out],out,kind="matrix",needs=[run],model=MODEL,cells=1,slices=[("tower",27)],grid=11,images=13000)
    for comparison in ["projector","sibling"]:
        for stage in (["projected"] if comparison=="projector" else STAGES):
            for task in ["recognition","depth","normal","kitti"]:
                first=runpath("recognition" if task=="recognition" else "kitti" if task=="kitti" else "geometry",MODEL,"pool4" if task=="recognition" else "full")
                second=first if comparison=="projector" else runpath("recognition" if task=="recognition" else "kitti" if task=="kitti" else "geometry","qwen3_5","pool4" if task=="recognition" else "full")
                if task=="kitti" and stage=="merged": continue
                if comparison=="projector" and task in ["depth","normal"]: continue
                out=output("bootstrap",f"{task}-{MODEL}-{comparison}-{stage}")
                argv=[P,"-m","scripts.semantic_bootstrap" if task=="recognition" else "scripts.geometry_bootstrap","--first-run",first,"--first-name",MODEL+"-"+stage,"--first-stage",stage,"--first-layer","27","--second-run",second,"--second-name",("tower" if comparison=="projector" else "qwen3_5-"+stage),"--second-stage","tower" if comparison=="projector" else stage,"--second-layer","27","--device","cuda:0","--out",out]
                argv+=["--labels",D/"imagenet100/validation_labels.json"] if task=="recognition" else ["--targets",D/("kitti" if task=="kitti" else "diode")/"val_targets.npz","--task","depth" if task=="kitti" else task]
                add(f"bootstrap-{comparison}-{task}-{stage}",argv,out,needs=[first,second])
    first=output("perturbation",MODEL+"_perturbation")
    out=output("bootstrap","perturbation-"+MODEL+"-projected-vs-tower")
    add("bootstrap-projector-perturbation",[P,"-m","scripts.cross_run_bootstrap","perturbation","--first",first,"--first-name","projected","--second",first,"--second-name","tower","--stage","projected","--second-stage","tower","--out",out],out,needs=[first])
    for mode in ["audit","sensitivity"]:
        for model in [MODEL,"qwen3_5"]:
            primary=output("perturbation",model+"_perturbation")
            audit=output("diagnostics/qwen38-flash-next-conditioning","conditioning-"+model+"-audit")
            out=output("diagnostics/qwen38-flash-next-conditioning","conditioning-"+model+"-"+mode)
            argv=[P,"-m","scripts.qwen_conditioning","--mode",mode,"--clean-run",E/"features/perturbation/identity"/f"{model}_448_pool4","--features-root",E/"features/perturbation","--model-dir",f"{model}_448_pool4","--labels",D/"imagenet100/validation_labels.json","--transform-manifest",D/"perturbation/transforms.manifest.json","--primary",primary,"--out",out]
            add("conditioning-"+mode+"-"+model,argv,out,kind="perturbation" if mode=="sensitivity" else "json",needs=[primary]+([audit] if mode=="sensitivity" else []),model=model,**({"cells":3} if mode=="sensitivity" else {}))
    diagnostic="diagnostics/qwen38-flash-next-conditioning"
    first=output(diagnostic,"conditioning-"+MODEL+"-sensitivity")
    second=output(diagnostic,"conditioning-qwen3_5-sensitivity")
    for stage in ["projector",*STAGES]:
        within=stage=="projector"
        other=first if within else second
        out=output(diagnostic,"paired-"+("projector-vs-tower" if within else "flash-next-vs-27b-"+stage))
        add("conditioning-paired-"+stage,[P,"-m","scripts.cross_run_bootstrap","perturbation","--first",first,"--first-name","projected" if within else MODEL,"--second",other,"--second-name","tower" if within else "qwen3_5","--stage","projected" if within else stage,"--second-stage","tower" if within else stage,"--out",out],out,needs=[first,other])
    out=output(diagnostic,"channel-details")
    add("conditioning-channels",[P,"-m","scripts.qwen_conditioning_channels","--features-root",E/"features/perturbation/identity","--out",out],out,needs=[first,second])
    for stage in STAGES:
        first=output("perturbation",MODEL+"_perturbation");second=output("perturbation","qwen3_5_perturbation")
        out=output("bootstrap","perturbation-flash-next-vs-27b-"+stage)
        add("bootstrap-sibling-perturbation-"+stage,[P,"-m","scripts.cross_run_bootstrap","perturbation","--first",first,"--first-name",MODEL,"--second",second,"--second-name","qwen3_5","--stage",stage,"--out",out],out,needs=[first,second])
    assert len({j["name"] for j in jobs})==len(jobs),"duplicate job names"
    assert len({j["output"] for j in jobs})==len(jobs),"duplicate job outputs"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def finite(value):
    if isinstance(value,float): assert math.isfinite(value),value
    elif isinstance(value,dict):
        for v in value.values(): finite(v)
    elif isinstance(value,list):
        for v in value: finite(v)

def check_intervals(value):
    if isinstance(value,dict):
        if "point_estimate" in value and "lower" in value and "upper" in value:
            assert value["confidence"]==0.95 and value["resamples"]==10000 and value["bootstrap_seed"]==0
            assert value["lower"]<=value["upper"]
        for v in value.values():check_intervals(v)
    elif isinstance(value,list):
        for v in value:check_intervals(v)

def validate(job):
    path=Path(job["output"]); expected=job["expect"]
    if job["kind"]=="perturbation-cache":
        protocol=json.loads((D/"perturbation/transforms.manifest.json").read_text())
        hashes={}
        for condition in protocol["conditions"]:
            image_root=D/"imagenet100/validation" if condition=="identity" else D/"perturbation"/condition
            subjob=dict(job,kind="cache",output=str(path.parent.parent/condition/path.name),
                        expect=dict(expected,images=str(image_root),limit=protocol["images"]))
            hashes.update(validate(subjob))
        return hashes
    if job["kind"]=="cache":
        import torch
        from vtb import cache
        assert set(cache.slices(path))=={tuple(s) for s in expected["slices"]}
        ids_expected=[p.stem for p in sorted(Path(expected["images"]).rglob("*")) if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
        ids_expected=ids_expected[:expected.get("limit")]
        hashes={}
        for stage,layer in expected["slices"]:
            ids=[]
            for shard in cache.shards(path,stage,layer):
                from safetensors import safe_open
                with safe_open(shard,framework="pt") as f:
                    meta=f.metadata();tokens=f.get_tensor("tokens")
                    assert meta["model_id"]==MID[expected["model"]]
                    assert int(meta["resolution"])==448 and meta["pooled_to"]==str(expected["pool"] or "none")
                    assert torch.isfinite(tokens).all()
                    width={"tower":1152,"merged":4608,"projected":2560 if expected["model"]==MODEL else 5120}[stage]
                    assert tokens.shape[1:]==(16 if expected["pool"] else 784 if stage=="tower" else 196,width)
                    ids+=meta["image_ids"].split("\n")
                hashes[str(shard)]=sha(shard)
            assert ids==ids_expected and len(ids)==len(set(ids))
        return hashes
    payload=json.loads(path.read_text())
    if job["kind"]=="correspondence" and expected["dataset"]=="spair":
        finite({k:v for k,v in payload.items() if k!="measurements"})
    else:
        finite(payload)
    check_intervals(payload)
    if "cells" in expected:
        assert len(payload["cells"])==expected["cells"]
    if job["kind"]=="matrix":
        assert payload["images"]==expected["images"]
        assert {(c["stage"],c["layer_index"]) for c in payload["cells"]}=={tuple(x) for x in expected["slices"]}
        for cell in payload["cells"]:
            assert cell["model_id"]==MID[expected["model"]] and cell["seeds"]==3
            from vtb.probe_run import LEARNING_RATES as SEMANTIC_GRID
            from vtb.geometry_run import LEARNING_RATES as GEOMETRY_GRID
            grid=SEMANTIC_GRID if expected["grid"]==11 else GEOMETRY_GRID
            assert {float(r) for r in cell["learning_rate_search"]}==set(grid)
            assert cell["learning_rate"] in grid
            assert abs(cell["relative_depth"]-cell["layer_index"]/27)<1e-4
            native={"tower":1152,"merged":4608,"projected":2560}[cell["stage"]]
            geometry="vtb.geometry_run" in job["argv"]
            matched="--match-capacity" in job["argv"]
            assert cell["token_width"]==(512 if geometry and matched else native)
            if geometry:
                assert payload["task"]==job["argv"][job["argv"].index("--task")+1]
                assert payload["epochs"]==10 and payload["seeds"]==3 and payload["head"]=="multiscale"
                assert payload["capacity_matched"]==matched and set(payload["learning_rates"])==set(GEOMETRY_GRID)
                assert payload["train_images"]==int(.7*expected["images"])
                assert cell["grid"]==(28 if cell["stage"]=="tower" else 14)
                assert len(cell["per_seed"])==3
                test_images=expected["images"]-int(.7*expected["images"])-int(.15*expected["images"])
                assert cell["images"]==test_images and all(seed["images"]==test_images for seed in cell["per_seed"])
            else:
                readout=job["argv"][job["argv"].index("--readout")+1]
                width=512 if matched else native
                parameters=100*width+100 if readout=="mean" else 1103460+1024*width
                assert cell["trainable_parameters"]==parameters
                fraction=float(job["argv"][job["argv"].index("--label-fraction")+1]) if "--label-fraction" in job["argv"] else 1.0
                assert payload["label_fraction"]==cell["label_fraction"]==fraction
                assert payload["full_train_images"]==cell["full_train_images"]==9100
                assert payload["train_images"]==cell["train_images"]==max(100,round(9100*fraction))
    if job["kind"]=="correspondence":
        rows=payload["measurements"]
        expected_cells={f"{stage}:27" for stage in STAGES}
        assert {(c["stage"],c["layer_index"]) for c in payload["cells"]}=={(stage,27) for stage in STAGES}
        assert all(set(row["cells"])==expected_cells for row in rows)
        assert payload["resolution"]==448 and payload["model_id"]==MID[expected["model"]]
        assert payload["pairs"]==len(rows)==len({r["pair_id"] for r in rows}) and len(rows)>0
        audit=json.loads((E/"pair-selection.json").read_text())["datasets"][expected["dataset"]]
        assert audit["pairs"]==len(rows)
        assert [r["pair_id"] for r in audit["rows"]]==[r["pair_id"] for r in rows]
        excluded=[]
        for actual,source in zip(rows,audit["rows"]):
            for key,metrics in actual["cells"].items():
                if expected["dataset"]=="spair":
                    assert metrics["keypoints"]==source["valid_keypoints"]
                    if source["valid_keypoints"]==0:
                        assert math.isnan(metrics["pck_0.1"])
                        excluded.append(actual["pair_id"])
                    else:
                        finite(metrics)
                else:
                    finite(metrics)
        if expected["dataset"]=="spair":
            assert set(excluded)==set(audit["zero_visible_keypoint_ids"])
        (E/(job["name"]+"-exclusions.json")).write_text(json.dumps({"selected_pairs":len(rows),"excluded_ids":sorted(set(excluded)),"reason":"zero source-visible keypoints after canonical crop" if excluded else None},indent=2)+"\n")
    if job["kind"]=="throughput":
        assert payload["images"]>=768 and payload["images_per_second"]>0
    splits=json.loads((ROOT/"results/evidence/qwen38-flash-next/splits.json").read_text())["datasets"]
    if job["kind"]=="perturbation":
        assert payload["images"]==2000 and payload["protocol"]["seeds"]==3
        assert payload["protocol"]["epochs"]==20 and payload["protocol"]["capacity_matched_width"]==512
        assert {(c["stage"],c["layer_index"]) for c in payload["cells"]}=={(stage,27) for stage in STAGES}
        required=set(json.loads((D/"perturbation/transforms.manifest.json").read_text())["conditions"])
        for cell in payload["cells"]:
            assert cell["model_id"]==MID[expected["model"]] and cell["readout"]=="attention" and cell["capacity_matched"]
            assert (cell["train_images"],cell["validation_images"],cell["test_images"])==(1400,300,300)
            assert {float(r) for r in cell["learning_rate_search"]}==set(payload["protocol"]["learning_rates"])
            assert len(cell["learning_rate_search"])==11 and cell["learning_rate"] in payload["protocol"]["learning_rates"]
            assert len(cell["conditions"])==13 and {c["condition"] for c in cell["conditions"]}==required
            assert set(cell["measurements"])==required-{"identity"}
            summaries={c["condition"]:c for c in cell["conditions"]}
            clean_reference=None
            for condition,rows in cell["measurements"].items():
                assert [r["image_id"] for r in rows]==splits["perturbation"]["test_ids"]
                assert all(len(r["clean"])==len(r["perturbed"])==3 for r in rows)
                assert all(v in [0,1] for r in rows for k in ["clean","perturbed"] for v in r[k])
                clean=[r["clean"] for r in rows]
                if clean_reference is None:clean_reference=clean
                assert clean==clean_reference
                for key,summary in [("clean",summaries["identity"]),("perturbed",summaries[condition])]:
                    by_seed=[sum(r[key][seed] for r in rows)/300 for seed in range(3)]
                    assert len(summary["accuracy_by_seed"])==3
                    assert all(abs(x-y)<1e-6 for x,y in zip(by_seed,summary["accuracy_by_seed"]))
                    assert abs(sum(by_seed)/3-summary["accuracy"])<1e-6
                point=sum(sum(r["clean"])-sum(r["perturbed"]) for r in rows)/900
                assert abs(point-summaries[condition]["clean_minus_condition"]["point_estimate"])<1e-6
    if isinstance(payload.get("first"),dict) and isinstance(payload.get("second"),dict) and "protocol" in payload:
        first,second=payload["first"],payload["second"]
        assert first["layer_index"]==second["layer_index"]==27
        if "predictions" in payload:
            rows=payload["predictions"]
            assert [r["image_id"] for r in rows]==splits["imagenet100"]["test_ids"]
            labels=json.loads((D/"imagenet100/validation_labels.json").read_text())["labels"]
            assert all(r["label"]==labels[r["image_id"]] for r in rows)
            for side in [first,second]:
                assert len(side["seed_accuracies"])==3
                assert len(side["learning_rate_search"])==11
            a,b=first["name"],second["name"]
            for row in rows: assert len(row[a])==len(row[b])==3
            assert all(isinstance(v,int) and 0<=v<100 for row in rows for key in [a,b] for v in row[key])
            for key,side in [(a,first),(b,second)]:
                by_seed=[sum(int(row[key][seed]==row["label"]) for row in rows)/len(rows) for seed in range(3)]
                assert all(abs(x-y)<1e-6 for x,y in zip(by_seed,side["seed_accuracies"]))
                assert abs(sum(by_seed)/3-side["mean_accuracy"])<1e-6
            difference=sum(sum(int(x==r["label"]) for x in r[a])-sum(int(x==r["label"]) for x in r[b]) for r in rows)/(3*len(rows))
            assert abs(difference-payload["difference"]["point_estimate"])<1e-6
        elif "measurements" in payload:
            rows=payload["measurements"]
            dataset="kitti" if "kitti" in job["name"] else "diode"
            protocol=payload["protocol"]
            assert protocol["epochs"]==10 and protocol["seeds"]==3 and protocol["head"]=="multiscale"
            assert protocol["task"]==job["argv"][job["argv"].index("--task")+1]
            arm=job["argv"][job["argv"].index("--arm")+1] if "--arm" in job["argv"] else "matched"
            assert protocol["arm"]==arm and set(protocol["learning_rates"])=={.0001,.0003,.001,.003,.01,.03}
            scenes=json.loads((D/dataset/"val_manifest.json").read_text()).get("scenes",{})
            assert all(row["scene"]==scenes.get(row["image_id"],"unknown") for row in rows)
            assert [r["image_id"] for r in rows]==splits[dataset]["test_ids"]
            for side in [first,second]:
                assert len(side["seed_metrics"])==3
                assert len(side["learning_rate_search"])==6
            a,b=first["name"],second["name"]
            for row in rows: assert len(row[a])==len(row[b])==3
            import torch
            values={key:torch.tensor([[row[key][seed] for row in rows] for seed in range(3)],dtype=torch.float32) for key in [a,b]}
            for key,side in [(a,first),(b,second)]:
                by_seed=values[key].mean(1).tolist()
                assert all(abs(x-y)<1e-6 for x,y in zip(by_seed,side["seed_metrics"]))
                assert abs(values[key].mean(1).mean().item()-side["mean_metric"])<1e-6
            difference=(values[a].mean(0)-values[b].mean(0)).mean().item()
            if not payload["protocol"]["higher_is_better"]: difference=-difference
            assert abs(difference-payload["first_advantage"]["point_estimate"])<1e-6
    if len(job["argv"])>3 and job["argv"][2]=="scripts.cross_run_bootstrap" and job["argv"][3]=="perturbation":
        a=job["argv"][job["argv"].index("--first-name")+1]
        b=job["argv"][job["argv"].index("--second-name")+1]
        required=set(json.loads((D/"perturbation/transforms.manifest.json").read_text())["conditions"])-{"identity"}
        assert set(payload["conditions"])==required
        expected_ids=set(splits["perturbation"]["test_ids"])
        for condition,summary in payload["conditions"].items():
            rows=summary["measurements"]
            assert len(rows)==len(expected_ids)==summary["test_images"]==300
            assert {r["image_id"] for r in rows}==expected_ids
            assert all(len(r[a])==len(r[b])==len(r[a+"_clean"])==len(r[b+"_clean"])==3 for r in rows)
            point=sum(sum(r[a])-sum(r[b]) for r in rows)/(3*len(rows))
            clean=sum(sum(r[a+"_clean"])-sum(r[b+"_clean"]) for r in rows)/(3*len(rows))
            assert abs(point-summary["difference"]["point_estimate"])<1e-6
            assert abs(clean-payload["clean"]["difference"]["point_estimate"])<1e-6
    if len(job["argv"])>2 and job["argv"][2]=="scripts.qwen_conditioning":
        mode=job["argv"][job["argv"].index("--mode")+1]
        primary=Path(job["argv"][job["argv"].index("--primary")+1])
        assert payload["provenance"]["primary_sha256_before_and_after"]==sha(primary)
        assert payload["provenance"]["device"]=="cpu" and payload["provenance"]["torch_threads"]==6
        stats=payload["statistics"] if mode=="audit" else payload["conditioning_statistics"]
        assert len(stats)==3 and {r["stage"] for r in stats}==set(STAGES)
        assert all(r["fit_images"]==1400 and r["tokens_per_image"]==16 and r["reduced_width"]==512 for r in stats)
        logs=payload["training_losses"]
        assert len(logs)==(3 if mode=="audit" else 42)
        for log in logs:
            assert log["epochs"]==20 and log["batch_size"]==256 and log["train_images"]==1400
            assert len(log["epoch_losses"])==20 and len(log["step_losses"])==120
            assert sum(v["images"] for v in log["step_losses"])==28000
            assert log["nonfinite_loss_steps"]==sum(v["loss"] is None for v in log["step_losses"])
            if mode=="sensitivity" and log["phase"]=="selected":assert log["nonfinite_loss_steps"]==0
        if mode=="audit":
            assert all(abs(r["primary_seed0_replay_accuracy"]-r["primary_recorded_seed0_accuracy"])<1e-6 for r in stats)
        else:
            assert all(r["standardized"] for r in stats)
            for stage in STAGES:
                selected=[r for r in logs if r["stage"]==stage and r["phase"]=="selected"]
                searched=[r for r in logs if r["stage"]==stage and r["phase"]=="search"]
                assert [r["seed"] for r in selected]==[0,1,2]
                assert len(searched)==11 and all(r["seed"]==0 for r in searched)
    if len(job["argv"])>2 and job["argv"][2]=="scripts.qwen_conditioning_channels":
        assert len(payload["rows"])==6 and payload["std_correction"]==1 and payload["minimum_scale"]==1e-6
        assert payload["device"]=="cpu" and payload["torch_threads"]==6
        assert {(r["model"],r["stage"]) for r in payload["rows"]}=={(m,s) for m in [MODEL,"qwen3_5"] for s in STAGES}
        for row in payload["rows"]:
            assert row["fit_images"]==1400 and row["tokens_per_image"]==16
            assert row["fit_image_ids"]==splits["perturbation"]["train_ids"]
            width={"tower":1152,"merged":4608,"projected":2560 if row["model"]==MODEL else 5120}[row["stage"]]
            assert all(len(row[k])==width for k in ["mean_fp32","sample_std_fp32","scale_fp32","rms_fp64"])
            assert all(scale==max(std,1e-6) or abs(scale-max(std,1e-6))<1e-12 for scale,std in zip(row["scale_fp32"],row["sample_std_fp32"]))
            audit=json.loads(output("diagnostics/qwen38-flash-next-conditioning","conditioning-"+row["model"]+"-audit").read_text())
            stat=next(r for r in audit["statistics"] if r["stage"]==row["stage"])
            assert abs(max(row["rms_fp64"])-stat["channel_rms_max"])<1e-10
    if "excluded_nonfinite_pair_ids" in payload:
        source_paths=[Path(x) for x in job["needs"] if str(x).endswith(".json")]
        if source_paths:
            originals=[json.loads(p.read_text()) for p in source_paths]
            dataset=originals[0].get("dataset")
            if dataset:
                audit=json.loads((E/"pair-selection.json").read_text())["datasets"][dataset]
                expected_excluded=set(audit.get("zero_visible_keypoint_ids",[]))
                actual_excluded=set(payload["excluded_nonfinite_pair_ids"])
                assert actual_excluded==expected_excluded
                rows=payload.get("measurements",[])
                assert len(rows)==audit["pairs"]-len(expected_excluded)
                if "first_advantage" in payload:
                    point=sum(r["first"]-r["second"] for r in rows)/len(rows)
                    assert abs(point-payload["first_advantage"]["point_estimate"])<1e-6
                if "finite_pairs" in payload: assert payload["finite_pairs"]==len(rows)
                assert {r["pair_id"] for r in rows}=={r["pair_id"] for r in audit["rows"]}-expected_excluded
    return {str(path):sha(path)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--gpu",choices=["0","1"])
    ap.add_argument("--cpu",action="store_true")
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--only",default="")
    ap.add_argument("--phase",choices=["geometry-kitti"])
    args=ap.parse_args();build()
    (E/"suite-plan.json").write_text(json.dumps(jobs,indent=2)+"\n")
    if not args.execute: print(len(jobs),"jobs planned");return
    assert args.cpu or args.gpu is not None,"explicit GPU reservation required"
    parity=json.loads((E/"parity-gpu-bf16.json").read_text())
    assert parity["source_repo"]=="Qwen/Qwen3.8-Flash-Next"
    assert parity["source_revision"]=="de4b8e4d43b917e7706784d8bb445c9af86a3540"
    assert parity["reference_class"]=="Qwen4ExpVisionModel" and parity["dtype"]=="torch.bfloat16"
    assert len(parity["identity"])==333
    assert parity["identity"]["merger.linear_fc2.weight"]["shape"]==[2560,4608]
    assert all(v["dtype"]=="torch.bfloat16" and len(v["sha256"])==64 for v in parity["identity"].values())
    assert len(parity["cases"])==20
    assert {(c["resolution"],c["stage"],c["layer"]) for c in parity["cases"]}=={(r,s,l) for r in [96,448] for s,l in SLICES}
    assert all(c["bit_exact"] and c["max_abs"]==0 for c in parity["cases"])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="" if args.cpu else args.gpu,HF_HOME="/tmp/vtb-delivery/hf",OMP_NUM_THREADS="6" if args.cpu else str(args.threads),MKL_NUM_THREADS="6" if args.cpu else str(args.threads),GPUS="" if args.cpu else args.gpu,LANES="1",THREADS="6" if args.cpu else str(args.threads))
    source_hashes={str(p):sha(p) for base in ["vtb","scripts"] for p in (ROOT/base).rglob("*.py")}
    source_id=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    completed={}
    for line in (E/"execution.jsonl").read_text().splitlines():
        r=json.loads(line)
        if r.get("exit_code")==0:
            previous=r.get("source_hashes",{})
            relevant={k:v for k,v in source_hashes.items() if "/vtb/" in k}
            argv=r.get("argv",[])
            if len(argv)>2 and argv[1]=="-m":
                module_path=str(ROOT/(argv[2].replace(".","/")+".py"))
                if module_path in source_hashes:relevant[module_path]=source_hashes[module_path]
            if previous and all(previous.get(k)==v for k,v in relevant.items()):
                completed[r.get("job")]=r
    for job in jobs:
        if (E/"PAUSE").exists():
            print("PAUSED at job boundary",flush=True);break
        if args.only and args.only not in job["name"]:continue
        if args.phase=="geometry-kitti" and not (
            "/measurements/geometry/" in job["output"] or "/measurements/kitti/" in job["output"]
            or (len(job["argv"])>2 and job["argv"][2]=="scripts.geometry_bootstrap")
        ):continue
        if args.cpu:
            job=dict(job,argv=["cpu" if x=="cuda:0" else x for x in job["argv"]])
        out=Path(job["output"]);out.parent.mkdir(parents=True,exist_ok=True)
        if job["name"] in completed:
            try:
                if validate(job)==completed[job["name"]]["output_hashes"]: continue
            except Exception:pass
        missing=[p for p in job["needs"] if not Path(p).exists()]
        producers={j["output"]:j for j in jobs}
        for dependency in job["needs"]:
            producer=producers.get(dependency)
            if producer and Path(dependency).exists():
                if producer["kind"] in ["cache","perturbation-cache"]:
                    if producer["name"] not in completed:missing.append("unverified cache "+dependency)
                else:
                    try:validate(producer)
                    except Exception as error:missing.append("invalid producer "+dependency+": "+repr(error))
        if missing:
            print("BLOCKED",job["name"],missing,flush=True);continue
        start=time.monotonic();record=dict(job=job["name"],started_at=datetime.datetime.now(datetime.UTC).isoformat(),argv=job["argv"],source_id=source_id,source_hashes=source_hashes,environment={k:env[k] for k in ["CUDA_VISIBLE_DEVICES","HF_HOME","OMP_NUM_THREADS","MKL_NUM_THREADS","GPUS","LANES","THREADS"]})
        record["runner_sha256"]=sha(Path(__file__))
        record["runner_source"]=str(Path(__file__))
        record["gpu_lock"]=None if args.cpu else f"/tmp/vtb-delivery/locks/gpu{args.gpu}.lock"
        record["cpu_hardware_sha256"]=sha(E/"cpu-hardware.json")
        logfile=E/(job["name"]+".log");record["log"]=str(logfile)
        record["input_manifest_hashes"]={str(p):sha(p) for p in [D/"imagenet100/validation_labels.json",D/"diode/val_manifest.json",D/"kitti/val_manifest.json",D/"perturbation/transforms.manifest.json"]}
        with (E/"execution.jsonl").open("a") as f:f.write(json.dumps(record|{"status":"started"})+"\n")
        lock_context=nullcontext() if args.cpu else open(f"/tmp/vtb-delivery/locks/gpu{args.gpu}.lock","w")
        with lock_context as lock,logfile.open("w") as log:
            if lock is not None: fcntl.flock(lock,fcntl.LOCK_EX)
            print("START",job["name"],flush=True)
            code=subprocess.run(["bash","-ic",shlex.join(job["argv"])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=() if lock is None else (lock.fileno(),)).returncode
        record["process_exit_code"]=code
        record["exit_code"]=code;record["elapsed_seconds"]=time.monotonic()-start
        record["ended_at"]=datetime.datetime.now(datetime.UTC).isoformat()
        if code==0:
            try:record["output_hashes"]=validate(job)
            except Exception as error:record["exit_code"]=1;record["validation_error"]=repr(error)
        with (E/"execution.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
        if record["exit_code"]==0:completed[job["name"]]=record
        print("END",job["name"],record["exit_code"],flush=True)

if __name__=="__main__":main()

