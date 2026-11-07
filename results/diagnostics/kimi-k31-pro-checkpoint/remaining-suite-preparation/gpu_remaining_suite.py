"""Run remaining complete matrices only under a newly recorded exclusive GPU grant."""

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/tmp/vtb-delivery/issues/24")
PUBLIC = Path("results/diagnostics/kimi-k31-pro-checkpoint")
MODELS = {"kimi_k31_pro": "immanuelpeter/MoonViT-K3.1-Pro", "moonvit_v2": "immanuelpeter/MoonViT-V2"}
RELEASES = {"kimi_k31_pro": "6ab95e2daf86e8100be2e712a7a71c0fba26dac4", "moonvit_v2": "bcaedfccd1f07d23ab20de8a0a168668c27cacb3"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def jobs():
    rows = []
    for task in ("depth", "normal"):
        for arm in ("raw", "matched"):
            for model in MODELS:
                out = ROOT / "measurements/geometry" / f"{model}_geometry_{task}_{arm}.json"
                argv = [".venv/bin/python", "-m", "vtb.geometry_run", "--run", str(ROOT / f"features/cpu4-bf16-bs1/geometry/{model}_448_full"),
                        "--targets", "/tmp/vtb-delivery/data/diode/val_targets.npz", "--task", task, "--out", str(out), "--device", "cuda:0"]
                if arm == "matched":
                    argv += ["--match-capacity"]
                rows.append(dict(kind="geometry", model=model, task=task, arm=arm, out=str(out), argv=argv, primary_cells=10))
    plan = read(ROOT / "fullsuite-plan.json")
    for job in plan["jobs"]:
        if job["phase"] == "transfer":
            rows.append(dict(kind="kitti", model=job["name"], task="depth", arm="matched", out=job["argv"][job["argv"].index("--out") + 1], argv=job["argv"], primary_cells=2))
        elif job["phase"] == "paired-fits" and "scripts.geometry_bootstrap" in job["argv"]:
            index = int(job["name"].split("-")[-1])
            rows.append(dict(kind="geometry-pair", paired_index=index, argv=job["argv"], out=job["argv"][job["argv"].index("--out") + 1], paired_cells=1))
        elif job["phase"] == "correspondence" and not job["name"].endswith("-navi"):
            model, dataset = job["name"].rsplit("-", 1)
            argv = list(job["argv"])
            argv[1:3] = [str(ROOT / "pinned_kimi_correspondence.py")]
            argv += ["--evaluation-side", "64", "--num-correspondences", "1000"]
            rows.append(dict(kind="correspondence", model=model, dataset=dataset, argv=argv, out=argv[argv.index("--out") + 1], primary_cells=3))
        elif job["phase"] == "throughput":
            rows.append(dict(kind="throughput", model=job["name"], argv=job["argv"], out=job["argv"][job["argv"].index("--out") + 1], primary_cells=1))
    assert len(rows) == 43 and sum(row.get("primary_cells", 0) for row in rows) == 98
    assert sum(row.get("paired_cells", 0) for row in rows) == 27
    return rows


def validate(row):
    import torch
    from scripts.validate_kimi_results import validate_probe, validate_correspondence, finite_numbers, GEOMETRY_RATES
    from vtb.geometry_bootstrap import paired_metric_bootstrap

    path, kind = Path(row["out"]), row["kind"]
    payload = read(path)
    if kind in {"geometry", "kitti"}:
        dataset = "diode" if kind == "geometry" else "kitti"
        selected, splits = read(ROOT / f"ids-{dataset}.json")["ids"], read(ROOT / f"splits-{dataset}.json")
        audit = validate_probe(payload, MODELS[row["model"]], selected, kind, row["arm"], expected_splits=splits)
        assert payload["device"] == "cuda:0" and payload["batch_size"] == 8 and payload["head"] == "multiscale"
        assert payload["seeds"] == 3 and set(payload["learning_rates"]) == GEOMETRY_RATES
        feature = payload["feature_provenance"]["details"]
        assert feature["release_revision"] == RELEASES[row["model"]]
        assert feature["device"] == ("cpu" if kind == "geometry" else "cuda:0")
        for cell in payload["cells"]:
            assert ("d1" if row["task"] == "depth" else "mean_deg") in cell
            for metric in ("d1", "abs_rel", "rmse") if row["task"] == "depth" else ("mean_deg",):
                if metric in cell:
                    assert abs(cell[metric] - sum(seed[metric] for seed in cell["per_seed"]) / 3) < 1.1e-4
        return audit
    if kind == "geometry-pair":
        argv = row["argv"]
        get = lambda name: argv[argv.index(name) + 1]
        kitti = "/kitti/" in get("--targets")
        splits = read(ROOT / ("splits-kitti.json" if kitti else "splits-diode.json"))
        metric = "d1" if get("--task") == "depth" else "mean_deg"
        protocol = payload["protocol"]
        assert protocol["task"] == get("--task") and protocol["metric"] == metric
        assert protocol["arm"] == get("--arm") and protocol["head"] == "multiscale"
        assert protocol["epochs"] == 10 and protocol["seeds"] == 3 and set(protocol["learning_rates"]) == GEOMETRY_RATES
        assert [value["image_id"] for value in payload["measurements"]] == splits["test"]
        assert len(set(splits["test"])) == len(splits["test"]) == protocol["test_images"]
        assert finite_numbers(payload)
        tensors = []
        for side in ("first", "second"):
            cell = payload[side]
            name = get(f"--{side}-name")
            expected_model = name.split(":")[0]
            assert cell["model_id"] == MODELS[expected_model] and cell["name"] == name
            assert cell["stage"] == get(f"--{side}-stage") and cell["layer_index"] == int(get(f"--{side}-layer"))
            search = cell["learning_rate_search"]
            assert len(search) == 6 and {float(rate) for rate in search} == GEOMETRY_RATES
            best = max(search.values()) if get("--task") == "depth" else min(search.values())
            assert next(value for rate, value in search.items() if float(rate) == cell["learning_rate"]) == best == cell["validation_score"]
            tensor = torch.tensor([record[name] for record in payload["measurements"]]).T
            assert tensor.shape == (3, len(splits["test"])) and torch.isfinite(tensor).all()
            assert torch.equal(tensor.mean(1), torch.tensor(cell["seed_metrics"]))
            assert abs(tensor.mean(1).mean().item() - cell["mean_metric"]) < 1e-7
            tensors.append(tensor)
        expected = paired_metric_bootstrap(*tensors, higher_is_better=get("--task") == "depth")
        assert payload["first_advantage"] == expected
        for scene, interval in payload["by_scene"].items():
            positions = [i for i, value in enumerate(payload["measurements"]) if value["scene"] == scene]
            assert interval == paired_metric_bootstrap(*(tensor[:, positions] for tensor in tensors), higher_is_better=get("--task") == "depth") | {"test_images":len(positions)}
        return dict(passed=True, test_images=len(splits["test"]), seeds=3, learning_rates=6,
                    exact_overall_and_scene_bootstrap_replay=True, point_source="same paired refits and rows; not a matrix point substitution")
    runtime = read(path.with_suffix(".runtime.json"))
    assert len(runtime["adapters"]) == 1
    adapter = runtime["adapters"][0]
    assert adapter["checkpoint"]["model_id"] == MODELS[row["model"]]
    assert adapter["checkpoint"]["bundle_revision"] == RELEASES[row["model"]]
    assert adapter["identity_tensors"] == 168 and adapter["exact_immutable_identity"]
    assert adapter["device"] == "cuda:0" and adapter["dtype"] == "torch.bfloat16" and adapter["threads"] == 8
    assert adapter["cuda_visible_devices"] == os.environ["CUDA_VISIBLE_DEVICES"]
    if kind == "correspondence":
        dataset = row["dataset"]
        selected = read(ROOT / f"pairs-{dataset}.json")["ids"]
        audit = validate_correspondence(payload, selected, read(ROOT / "spair-visibility-audit.json") if dataset == "spair" else None)
        assert payload["model_id"] == MODELS[row["model"]] and payload["evaluation_side"] == 64 and payload["num_correspondences"] == 1000
        for cell in payload["cells"]:
            key = f"{cell['stage']}:27"
            for metric, value in cell.items():
                if metric in ("stage","layer_index","by_class","macro_pck_0.1"):
                    continue
                values = [record["cells"][key][metric] for record in payload["measurements"] if math.isfinite(record["cells"][key][metric])]
                assert values and abs(value - sum(values) / len(values)) < 1e-6
        assert adapter["validated_image_forwards"] == 2 * len(selected) and all(adapter["native_integrity"].values())
        return audit | {"runtime":runtime}
    assert payload["model_id"] == MODELS[row["model"]] and payload["images"] == 768
    assert payload["resolution"] == 448 and payload["batch_size"] == 1 and payload["workers"] == 2
    assert payload["configured_patch_size"] == 14 and payload["tower_tokens_per_image"] == 1024 and payload["warmup_batches"] == 2
    assert finite_numbers(payload) and payload["seconds"] > 0 and payload["images_per_second"] > 0
    assert abs(payload["images_per_second"] - 768 / payload["seconds"]) < .1
    selection = read(ROOT / "throughput-source-selection.json")
    assert selection["count"] == 768
    return dict(passed=True, images=768, batch_size=1, workers=2, runtime=runtime, actual_source_selection=selection)


def register(row, report):
    path = Path(row["out"])
    inventory = read(ROOT / "completed-inventory.json")
    expected = read(ROOT / "expected-inventory.json")
    payload = read(path)
    if row["kind"] == "geometry-pair":
        keys = [expected["paired_comparison_cells"][row["paired_index"]]]
        collection = "completed_paired"
        destination = Path("results/bootstrap")
    else:
        collection = "completed"
        model = row["model"]
        if row["kind"] in {"geometry", "kitti"}:
            keys = [dict(task=row["task"] if row["kind"] == "geometry" else "kitti", model=model,
                         stage=cell["stage"], layer=cell["layer_index"], arm=row["arm"]) for cell in payload["cells"]]
        elif row["kind"] == "correspondence":
            keys = [dict(task=row["dataset"], model=model, stage=cell["stage"], layer=cell["layer_index"]) for cell in payload["cells"]]
        else:
            keys = [dict(task="throughput", model=model)]
        folder = "correspondence" if row["kind"] == "correspondence" else row["kind"]
        destination = Path("results") / folder if model == "kimi_k31_pro" else PUBLIC / "sibling-control" / folder
    destination.mkdir(parents=True, exist_ok=True)
    for key in keys:
        records = [record for record in inventory[collection] if record["cell"] == key]
        if records:
            assert len(records) == 1 and records[0]["sha256"] == sha(path)
        else:
            inventory[collection].append(dict(cell=key, source=str(path), sha256=sha(path)))
    for collection, expected_key, completed_key in (("completed","cells","completed_primary_cells"),("completed_paired","paired_comparison_cells","completed_paired_cells")):
        actual = {json.dumps(value["cell"],sort_keys=True) for value in inventory[collection]}
        assert len(actual) == len(inventory[collection]) and actual <= {json.dumps(value,sort_keys=True) for value in expected[expected_key]}
        inventory[completed_key] = len(actual)
    actual = {json.dumps(value["cell"],sort_keys=True) for value in inventory["completed"]}
    inventory["pending"] = [value for value in expected["cells"] if json.dumps(value,sort_keys=True) not in actual]
    for target in (ROOT/"completed-inventory.json",PUBLIC/"completed-inventory.json"):
        target.write_text(json.dumps(inventory,indent=2)+"\n")
    for source in (path,path.with_suffix(".validation.json"),path.with_suffix(".runtime.json")):
        if source.exists():
            shutil.copy2(source,destination/source.name)
    if path.with_suffix(".log").exists():
        shutil.copy2(path.with_suffix(".log"),destination/path.with_suffix(".txt").name)


def source_preflight(row):
    if row["kind"] not in {"correspondence", "throughput"}:
        return
    prepared_sha = sha("/tmp/vtb-delivery/data-sha256.txt")
    if row["kind"] == "correspondence":
        import inspect
        from scripts.correspondence_run import DATASETS
        dataset = row["dataset"]
        selection = read(ROOT / f"pairs-{dataset}.json")
        proof_path = ROOT / f"{dataset}-gpu-source-selection.json"
        source = sha(inspect.getfile(DATASETS[dataset]))
        if proof_path.exists():
            proof = read(proof_path)
            assert proof["ids"] == selection["ids"] and proof["dataset_source_sha256"] == source
            assert proof["prepared_input_manifest_sha256"] == prepared_sha
        else:
            actual = DATASETS[dataset](Path(selection["root"]), 448)
            assert len(actual) == len(selection["ids"])
            ids = [actual[i]["id"] for i in range(len(actual))]
            assert ids == selection["ids"] and len(set(ids)) == len(ids)
            proof_path.write_text(json.dumps(dict(passed=True, root=selection["root"], ids=ids,
                count=len(ids), dataset_source_sha256=source, prepared_input_manifest_sha256=prepared_sha), indent=2)+"\n")
    else:
        from vtb.images import ImageFolder
        actual = ImageFolder(Path("/tmp/vtb-delivery/data/imagenet100/validation"), None, 768)
        ids = [path.stem for path in actual.paths]
        assert len(ids) == len(set(ids)) == 768
        proof = dict(passed=True, count=768, ids=ids, paths=[str(path) for path in actual.paths],
            loader_shuffle=False, warmup_batches=2, warmup_ids=ids[:2], batch_size=1,
            prepared_input_manifest_sha256=prepared_sha)
        target=ROOT/"throughput-source-selection.json"
        if target.exists():
            assert read(target) == proof
        else:
            target.write_text(json.dumps(proof,indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope",choices=("all","dense","correspondence-throughput"),default="all")
    parser.add_argument("--prepare-only",action="store_true")
    args = parser.parse_args()
    queue = jobs()
    if args.scope != "all":
        dense = {"geometry","kitti","geometry-pair"}
        queue = [row for row in queue if (row["kind"] in dense) == (args.scope == "dense")]
    if args.prepare_only:
        (ROOT/"remaining-gpu-jobs.json").write_text(json.dumps(dict(status="prepared only, no CUDA grant",jobs=queue,
            primary_cells=sum(row.get("primary_cells",0) for row in queue),paired_cells=sum(row.get("paired_cells",0) for row in queue),
            correspondence_cpu_comparisons_afterward=8,raw_attention_cpu_phase_separate=True),indent=2)+"\n")
        print(f"Prepared {len(queue)} jobs; CUDA not initialized")
        return
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    grant = read(ROOT/"gpu-grant.json")
    assert gpu in ("0","1") and grant["physical_gpu"] == int(gpu) and grant.get("active") is True
    assert grant.get("phase") in ("remaining-suite",f"remaining-{args.scope}")
    assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "8"
    prior = [json.loads(line) for line in (ROOT/"execution.jsonl").read_text().splitlines() if json.loads(line).get("phase")=="cpu8-raw-recognition"]
    assert prior and prior[-1]["event"] == "end" and prior[-1]["exit_code"] == 0, "Finish CPU training normally before CUDA"
    with Path(f"/tmp/vtb-delivery/locks/gpu{gpu}.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock,fcntl.LOCK_UN)
            raise RuntimeError("Invoke via run_logged.py --gpu under its full-process exclusive flock")
    import torch
    assert torch.get_num_threads() == 8
    hardware = subprocess.check_output(["bash","-ic",f"nvidia-smi -i {gpu} --query-gpu=uuid,name --format=csv,noheader"],text=True).strip()
    progress=[]
    for row in queue:
        out = Path(row["out"])
        out.parent.mkdir(parents=True,exist_ok=True)
        resumed=out.exists()
        began=time.monotonic()
        source_preflight(row)
        if not resumed:
            with out.with_suffix(".log").open("w") as log:
                process = subprocess.run(["bash","-ic",shlex.join(row["argv"])],env=os.environ|{"PYTHONPATH":".","GPUS":gpu,"LANES":"1","THREADS":"8"},stdout=log,stderr=subprocess.STDOUT)
            if process.returncode:
                raise SystemExit(process.returncode)
        report=validate(row)
        report.update(command=shlex.join(row["argv"]),elapsed_seconds=time.monotonic()-began,resumed=resumed,
                      actual_device="cuda:0",physical_gpu=int(gpu),gpu_identity=hardware,threads=8,validated_at=stamp(),
                      payload_sha256=sha(out),driver_sha256=sha(__file__),grant=grant)
        out.with_suffix(".validation.json").write_text(json.dumps(report,indent=2)+"\n")
        register(row,report)
        progress.append(dict(job=row,validation=report))
        (ROOT/"remaining-gpu-progress.json").write_text(json.dumps(dict(complete=False,results=progress),indent=2)+"\n")
        print(json.dumps(dict(kind=row["kind"],out=str(out),resumed=resumed,passed=True)),flush=True)
    (ROOT/"remaining-gpu-progress.json").write_text(json.dumps(dict(complete=True,scope=args.scope,results=progress),indent=2)+"\n")


if __name__ == "__main__":
    main()
