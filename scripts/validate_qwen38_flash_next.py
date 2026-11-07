"""Validate Qwen Flash-Next against its independent parent vision class on CPU or a reserved GPU."""
import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModel
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

from vtb.adapters.qwen3_5 import FLASH_NEXT, QWEN27B, Qwen38FlashNextAdapter, collate, Preprocess, source_json, source_weights


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def reference_raster(packed, batch, side):
    result = torch.empty(batch, side, side, packed.shape[-1], device=packed.device, dtype=packed.dtype)
    offset = 0
    for image in range(batch):
        for y in range(0, side, 2):
            for x in range(0, side, 2):
                for dy in range(2):
                    for dx in range(2):
                        result[image, y + dy, x + dx] = packed[offset]
                        offset += 1
    return result.flatten(1, 2)


def validate(bundle, device, resolutions=(96, 448), dtype=torch.float32):
    started = time.monotonic()
    weights = source_weights(FLASH_NEXT)
    state = load_file(bundle / "model.safetensors")
    assert set(weights) == set(state) and len(state) == 333
    identity = {}
    for name, expected in weights.items():
        actual = state[name]
        assert torch.equal(actual, expected), name
        identity[name] = {"shape": list(actual.shape), "dtype": str(actual.dtype), "sha256": digest(actual)}
    assert state["merger.linear_fc2.weight"].shape == (2560, 4608)
    config = Qwen4ExpVisionConfig(**source_json(FLASH_NEXT, "config.json")["vision_config"])
    reference = Qwen4ExpVisionModel._from_config(config, dtype=dtype).eval()
    reference.load_state_dict(weights, strict=True)
    reference = reference.to(device)
    source_file = Path(inspect.getfile(Qwen4ExpVisionModel))
    adapter = Qwen38FlashNextAdapter(resolution=resolutions[0], device=device, dtype=dtype)
    released = AutoModel.from_pretrained(bundle, local_files_only=True, dtype=dtype).to(device).eval()
    report = {"source_repo": FLASH_NEXT.repo, "source_revision": FLASH_NEXT.revision,
              "reference_class": "Qwen4ExpVisionModel", "reference_source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
              "reference_config": config.to_dict(), "torch": torch.__version__, "transformers": transformers.__version__,
              "dtype": str(dtype), "device": str(device), "identity": identity, "cases": []}
    rng = np.random.default_rng(22)
    images = [Image.fromarray(rng.integers(0,256,(317,503,3),dtype=np.uint8)),
              Image.fromarray(rng.integers(0,256,(401,299,3),dtype=np.uint8))]
    report["image_sha256"] = [hashlib.sha256(im.tobytes()).hexdigest() for im in images]
    for resolution in resolutions:
        adapter.resolution = resolution
        inputs = collate([Preprocess(resolution)(im) for im in images])
        captured, handles = {}, []
        def capture(layer):
            def hook(module, args, output):
                captured[layer] = output.detach()
            return hook
        for layer in adapter.depth_points():
            handles.append(reference.blocks[layer-1].register_forward_hook(capture(layer)))
        with torch.inference_mode():
            expected = reference(inputs["hidden_states"].to(device,dtype), inputs["grid_thw"].to(device))
            actual_release = released(inputs["hidden_states"].to(device,dtype), inputs["grid_thw"].to(device))
            batches = list(adapter.extract(inputs, ["fixed-22-0","fixed-22-1"]))
        for h in handles:
            h.remove()
        for actual, expected_tensor in [(actual_release.last_hidden_state,expected.last_hidden_state),(actual_release.pooler_output,expected.pooler_output)]:
            torch.testing.assert_close(actual,expected_tensor,rtol=0,atol=0)
        for batch in batches:
            if batch.stage == "tower":
                target = reference_raster(captured[batch.layer_index],2,resolution//16)
            elif batch.stage == "merged":
                target = expected.last_hidden_state.reshape(2,-1,4608)
            else:
                target = expected.pooler_output.reshape(2,-1,2560)
            target=target.cpu()
            delta=batch.tokens.float()-target.float()
            torch.testing.assert_close(batch.tokens,target,rtol=0,atol=0)
            report["cases"].append({"resolution":resolution,"stage":batch.stage,"layer":batch.layer_index,
                                    "shape":list(batch.tokens.shape),"max_abs":delta.abs().max().item(),"bit_exact":torch.equal(batch.tokens,target)})
    report["elapsed_seconds"]=time.monotonic()-started
    return report


def drift():
    first=source_weights(QWEN27B)
    second=source_weights(FLASH_NEXT)
    rows={}
    assert set(first)==set(second) and len(first)==333
    for name, value in first.items():
        other=second[name]
        row={"qwen27b_shape":list(value.shape),"flash_next_shape":list(other.shape)}
        if value.shape == other.shape:
            a,b=value.double().flatten(),other.double().flatten()
            row.update(cosine=torch.nn.functional.cosine_similarity(a,b,dim=0).item(),
                       relative_l2=((b-a).norm()/a.norm()).item(),equal=torch.equal(value,other))
        else:
            row["comparison"]="shape changed; cosine and relative L2 undefined"
        rows[name]=row
    assert sum("cosine" in r for r in rows.values()) == 331
    return {"first_revision":QWEN27B.revision,"second_revision":FLASH_NEXT.revision,"tensors":rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,default=FLASH_NEXT.bundle)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--dtype",choices=["float32","bfloat16"],default="float32")
    parser.add_argument("--resolutions",type=int,nargs="+",default=[96,448])
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--drift-out",type=Path)
    args=parser.parse_args()
    report=validate(args.bundle,args.device,args.resolutions,getattr(torch,args.dtype))
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    if args.drift_out:
        args.drift_out.write_text(json.dumps(drift(),indent=2)+"\n")


if __name__=="__main__":
    main()

