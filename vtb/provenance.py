import json
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger("vtb.provenance")


def recorded_source(bundle: Path) -> tuple[str, str] | None:
    manifest = bundle / "export_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        return data["source_repo"], data["source_revision"]
    config = bundle / "config.json"
    if config.is_file():
        data = json.loads(config.read_text())
        if "source" in data:
            return data["source"], data["revision"]
    return None


def check_provenance(bundle: Path, repo: str, revision: str) -> None:
    recorded = recorded_source(bundle)
    if recorded is not None and recorded != (repo, revision):
        raise ValueError(
            f"bundle provenance {recorded[0]}@{recorded[1]} does not match {repo}@{revision}"
        )


def log_loaded(model_id: str, origin: str) -> None:
    logger.info("%s loaded weights from %s", model_id, origin)


def open_release(
    model_id: str, source_repo: str, source_revision: str, *, bundle: Path | None = None
) -> Path:
    """Return a local bundle if given, else the latest Hub snapshot of our release, checked against its pinned parent."""
    release = bundle or Path(snapshot_download(model_id))
    check_provenance(release, source_repo, source_revision)
    log_loaded(
        model_id, f"{bundle or model_id} (source {source_repo}@{source_revision})"
    )
    return release
