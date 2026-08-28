"""Read selected tensors out of remote safetensors shards."""

import json
import time
from collections.abc import Iterable

import requests
import torch
from huggingface_hub import get_hf_file_metadata, hf_hub_url

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
ATTEMPTS = 6


def _range(url: str, start: int, end: int) -> bytes:
    """Read one byte span, backing off when the CDN throttles or fails."""
    for attempt in range(ATTEMPTS):
        response = requests.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=300)
        if attempt < ATTEMPTS - 1 and (response.status_code == 429 or response.status_code >= 500):
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
        return response.content


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


def read_header(repo: str, filename: str) -> tuple[dict, str, int]:
    """Return a shard's tensor index, its resolved URL, and where its data starts."""
    url = get_hf_file_metadata(hf_hub_url(repo, filename)).location
    size = int.from_bytes(_range(url, 0, HEADER_SIZE_BYTES), "little")
    header = json.loads(_range(url, HEADER_SIZE_BYTES, HEADER_SIZE_BYTES + size))
    header.pop("__metadata__", None)
    return header, url, HEADER_SIZE_BYTES + size


def load_prefixed(repo: str, filenames: Iterable[str], prefix: str) -> dict[str, torch.Tensor]:
    """Download only the tensors whose names start with ``prefix``."""
    weights = {}
    for filename in filenames:
        header, url, data_start = read_header(repo, filename)
        matches = [(name, entry) for name, entry in header.items() if name.startswith(prefix)]
        for span_start, span_end, members in _runs(matches):
            raw = _range(url, data_start + span_start, data_start + span_end)
            for name, entry in members:
                start, end = entry["data_offsets"]
                # frombuffer needs a writable copy, and safetensors stores row-major.
                chunk = bytearray(raw[start - span_start : end - span_start])
                flat = torch.frombuffer(chunk, dtype=DTYPES[entry["dtype"]])
                weights[name.removeprefix(prefix)] = flat.reshape(entry["shape"])
    return weights


def prefix_bytes(repo: str, filenames: Iterable[str], prefix: str) -> int:
    """Report what ``load_prefixed`` would download, without downloading it."""
    total = 0
    for filename in filenames:
        header, _, _ = read_header(repo, filename)
        for name, entry in header.items():
            if name.startswith(prefix):
                start, end = entry["data_offsets"]
                total += end - start
    return total
