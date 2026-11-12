"""Explicit readout methods and hash-bound compatibility for archived outputs."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

PRIMARY = "train-pca-v1"
STANDARDIZED = "train-channel-standardized-pca-v1"
RAW = "unreduced-v1"

def descriptor(*, standardize_channels=False, matched=True):
    if standardize_channels and not matched:
        raise ValueError("channel standardization requires matched capacity")
    return {"id": STANDARDIZED if standardize_channels else PRIMARY if matched else RAW,
            "fit_split": "train" if matched else None, "sample_std_correction": 1 if standardize_channels else None,
            "scale_floor": 1e-6 if standardize_channels else None,
            "svd_oversample": 16 if matched else None, "fit_seed": 0 if matched else None}

def payload_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()).hexdigest()

@lru_cache(maxsize=1)
def legacy_registry():
    path = Path(__file__).resolve().parents[1] / "results/method-provenance.json"
    return json.loads(path.read_text())["records"] if path.is_file() else {}

def identify(payload):
    method = payload.get("method", payload.get("protocol", {}).get("method"))
    if method is None:
        record = legacy_registry().get(payload_hash(payload), {})
        method = record.get("method")
    if not method or method.get("id") not in {PRIMARY, STANDARDIZED, RAW}:
        raise ValueError("unknown reducer method: explicit metadata or hash-bound provenance is required")
    expected = descriptor(standardize_channels=method["id"] == STANDARDIZED, matched=method["id"] != RAW)
    if method != expected:
        raise ValueError("inconsistent reducer method definition")
    return method

def require_same(first, second):
    a, b = identify(first), identify(second)
    if a != b:
        raise ValueError("reducer methods differ")
    return a
