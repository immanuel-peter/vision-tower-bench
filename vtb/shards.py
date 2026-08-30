"""Read selected tensors out of remote safetensors shards."""

import hashlib
import json
import os
import time
from collections.abc import Iterable
from pathlib import Path

import requests
import torch
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from safetensors.torch import load_file, save_file

DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
HEADER_SIZE_BYTES = 8

# One request per tensor gets rate limited, so neighbours closer than this ride along.
STRIDE_GAP_BYTES = 8 * 1024 * 1024
# Limit retry cost when the CDN resets a long transfer.
MAX_SPAN_BYTES = 256 * 1024 * 1024
ATTEMPTS = 6


def _get(url: str, start: int, end: int) -> bytes:
    """Read one byte span, backing off when the CDN throttles, fails, or drops the socket."""
    for attempt in range(ATTEMPTS):
        last = attempt == ATTEMPTS - 1
        try:
            response = requests.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=300)
        except requests.RequestException:
            if last:
                raise
            time.sleep(2**attempt)
            continue
        if not last and (response.status_code == 429 or response.status_code >= 500):
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
        return response.content
    raise RuntimeError(f"unreachable: {url} bytes {start} to {end}")


def _range(url: str, start: int, end: int) -> bytes:
    if end - start <= MAX_SPAN_BYTES:
        return _get(url, start, end)
    pieces = [
        _get(url, at, min(at + MAX_SPAN_BYTES, end))
        for at in range(start, end, MAX_SPAN_BYTES)
    ]
    return b"".join(pieces)


def _runs(entries: list[tuple[str, dict]]) -> list[tuple[int, int, list[tuple[str, dict]]]]:
    """Group tensors into byte spans that one request can cover."""
    spans: list[tuple[int, int, list[tuple[str, dict]]]] = []
    for name, entry in sorted(entries, key=lambda pair: pair[1]["data_offsets"][0]):
        start, end = entry["data_offsets"]
        if spans and start - spans[-1][1] <= STRIDE_GAP_BYTES:
            span_start, _, members = spans[-1]
            members.append((name, entry))
            spans[-1] = (span_start, end, members)
        else:
            spans.append((start, end, [(name, entry)]))
    return spans


def resolve(repo: str, filename: str, revision: str | None = None) -> tuple[str, str]:
    """Return a download URL and the etag identifying the revision behind it."""
    meta = get_hf_file_metadata(hf_hub_url(repo, filename, revision=revision))
    return meta.location, meta.etag or ""


def _header_at(url: str) -> tuple[dict, int]:
    size = int.from_bytes(_range(url, 0, HEADER_SIZE_BYTES), "little")
    header = json.loads(_range(url, HEADER_SIZE_BYTES, HEADER_SIZE_BYTES + size))
    header.pop("__metadata__", None)
    return header, HEADER_SIZE_BYTES + size


def read_header(repo: str, filename: str, revision: str | None = None) -> tuple[dict, str, int]:
    """Return a shard's tensor index, its resolved URL, and where its data starts."""
    url, _ = resolve(repo, filename, revision)
    header, data_start = _header_at(url)
    return header, url, data_start


def cache_root() -> Path:
    """Return the configured cache or a directory beside the Hugging Face cache."""
    if override := os.environ.get("VTB_SHARD_CACHE"):
        return Path(override)
    home = os.environ.get("HF_HOME")
    return (Path(home) if home else Path.home() / ".cache" / "huggingface") / "vtb-shards"


def cache_path(repo: str, prefix: str, etags: Iterable[tuple[str, str]]) -> Path:
    """Name a read by what it asks for and by the revision it would read."""
    identity = json.dumps([repo, prefix, sorted(etags)], sort_keys=True)
    return cache_root() / f"{hashlib.sha256(identity.encode()).hexdigest()[:32]}.safetensors"


def load_prefixed(
    repo: str,
    filenames: Iterable[str],
    prefix: str,
    *,
    revision: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load matching tensors and cache them locally."""
    filenames = list(filenames)
    resolved = {name: resolve(repo, name, revision) for name in filenames}
    path = cache_path(repo, prefix, ((name, etag) for name, (_, etag) in resolved.items()))
    if path.exists():
        return load_file(path)

    weights = {}
    for filename in filenames:
        url = resolved[filename][0]
        header, data_start = _header_at(url)
        matches = [(name, entry) for name, entry in header.items() if name.startswith(prefix)]
        for span_start, span_end, members in _runs(matches):
            raw = _range(url, data_start + span_start, data_start + span_end)
            for name, entry in members:
                start, end = entry["data_offsets"]
                # frombuffer needs a writable copy, and safetensors stores row-major.
                chunk = bytearray(raw[start - span_start : end - span_start])
                flat = torch.frombuffer(chunk, dtype=DTYPES[entry["dtype"]])
                weights[name.removeprefix(prefix)] = flat.reshape(entry["shape"])

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target so a killed process cannot leave a half-written cache entry.
    scratch = path.with_suffix(f".{os.getpid()}.partial")
    save_file(weights, scratch)
    scratch.replace(path)
    return weights


def prefix_bytes(
    repo: str,
    filenames: Iterable[str],
    prefix: str,
    *,
    revision: str | None = None,
) -> int:
    """Report what ``load_prefixed`` would download, without downloading it."""
    total = 0
    for filename in filenames:
        header, _, _ = read_header(repo, filename, revision)
        for name, entry in header.items():
            if name.startswith(prefix):
                start, end = entry["data_offsets"]
                total += end - start
    return total
