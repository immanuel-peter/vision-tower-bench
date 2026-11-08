"""Compare an owner's corrected published example with the bench on fresh files.

Prepared only. Requires a new explicit smoke reservation and inherited device
lock, after the original evaluation reaches an approved boundary.
"""

import argparse
import csv
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int, choices=(0, 1))
    parser.add_argument("--grant", required=True, type=Path)
    parser.add_argument("--lock-fd", type=int, default=9)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.physical_gpu)
    grant = json.loads(args.grant.read_text())
    assert grant["active"] and grant["physical_gpu"] == args.physical_gpu
    assert grant["phase"] == "corrected-published-example-smoke"
    assert os.readlink(f"/proc/self/fd/{args.lock_fd}") == f"/tmp/vtb-delivery/locks/gpu{args.physical_gpu}.lock"
    fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def query(command):
        return subprocess.check_output(["bash", "-ic", command], text=True,
                                       stderr=subprocess.DEVNULL).strip()

    uuid, memory = next(csv.reader([query(
        f"nvidia-smi -i {args.physical_gpu} --query-gpu=uuid,memory.used --format=csv,noheader,nounits")]))
    uuid, memory = uuid.strip(), int(memory.strip())
    processes = [line for line in query(
        "nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader").splitlines() if uuid in line]
    assert memory == 0 and not processes, (memory, processes)
    hardware_before = dict(physical_gpu=args.physical_gpu, gpu_uuid=uuid,
                           memory_mib=memory, cuda_processes=processes,
                           queried_before_torch_import=True)
    assert len(args.revision) == 40 and args.snapshot.name == args.revision
    snapshot = args.snapshot.absolute()
    script = snapshot / "examples/inference.py"
    assert script.is_file()
    candidate = Path("examples/kimi_k31_pro_inference.py")
    assert hashlib.sha256(script.read_bytes()).digest() == hashlib.sha256(candidate.read_bytes()).digest()

    import torch
    from PIL import Image, ImageOps
    from vtb.adapters.moonvit_v2 import K31_PRO, KimiK31ProAdapter

    torch.set_num_threads(2)
    captured = {}
    spec = importlib.util.spec_from_file_location("published_pro_example", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_loader = module.AutoModel.from_pretrained
    original_forward = module.Projector.forward
    handles = []

    def load(*positional, **kwargs):
        model = original_loader(*positional, **kwargs)

        def capture_tower(_module, _inputs, output):
            captured["tower"] = output.detach().unsqueeze(0).clone()

        handles.append(model.encoder.register_forward_hook(capture_tower))
        return model

    def project(self, tokens):
        captured["merged"] = tokens.detach().clone()
        output = original_forward(self, tokens)
        captured["projected"] = output.detach().clone()
        return output

    previous_argv = sys.argv
    try:
        module.AutoModel.from_pretrained = load
        module.Projector.forward = project
        sys.argv = [str(script), str(args.image)]
        module.main()
    finally:
        sys.argv = previous_argv
        module.AutoModel.from_pretrained = original_loader
        module.Projector.forward = original_forward
        for handle in handles:
            handle.remove()

    class FreshSnapshotAdapter(KimiK31ProAdapter):
        checkpoint = dataclasses.replace(K31_PRO, bundle_revision=None)
        model_id = str(snapshot)

    adapter = FreshSnapshotAdapter(resolution=448, device="cuda:0")
    with Image.open(args.image) as image:
        original_size = list(image.size)
        processed = adapter.preprocess()(ImageOps.exif_transpose(image).convert("RGB"))
    inputs = adapter.collate([processed])
    expected = {}
    with torch.inference_mode():
        for batch in adapter.extract(inputs, [args.image.stem]):
            if batch.layer_index == 27:
                expected[batch.stage] = batch.tokens
    assert set(captured) == set(expected) == {"tower", "merged", "projected"}
    results = {}
    for stage in expected:
        reference, actual = expected[stage], captured[stage]
        assert reference.dtype == actual.dtype == torch.bfloat16
        assert str(reference.device) == "cpu" and str(actual.device) == "cuda:0"
        torch.testing.assert_close(actual.cpu(), reference, atol=0, rtol=0)
        results[stage] = dict(shape=list(actual.shape), dtype=str(actual.dtype),
                              device=str(actual.device), max_abs=0, bit_exact=True)
    report = dict(passed=True, scope="corrected published example versus bench adapter on the same fresh snapshot; not official production SDPA parity",
                  revision=args.revision, snapshot=str(snapshot), physical_gpu=args.physical_gpu,
                  device="cuda:0", dtype="torch.bfloat16", batch_size=1, threads=2,
                  image=str(args.image), image_original_size=original_size,
                  image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                  comparison_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  hardware_before=hardware_before,
                  stages=results, official_production_bf16_parity_passed=False)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
