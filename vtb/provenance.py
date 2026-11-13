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


def check_provenance(bundle: Path, repo: str, revision: str, *, required: bool = False) -> None:
    recorded = recorded_source(bundle)
    if recorded is None:
        if required:
            raise ValueError(f"bundle {bundle} records no source provenance; expected {repo}@{revision}")
        return
    if recorded != (repo, revision):
        raise ValueError(f"bundle provenance {recorded[0]}@{recorded[1]} does not match {repo}@{revision}")


def log_loaded(model_id: str, origin: str) -> None:
    logger.info("%s loaded weights from %s", model_id, origin)


def open_release(model_id: str, revision: str, source_repo: str, source_revision: str, *, bundle: Path | None = None, required: bool = False) -> Path:
    """Return the release directory: a local bundle if given, else the Hub snapshot of model_id at revision."""
    origin = str(bundle) if bundle is not None else f"{model_id}@{revision}"
    bundle = Path(snapshot_download(model_id, revision=revision)) if bundle is None else bundle
    check_provenance(bundle, source_repo, source_revision, required=required)
    log_loaded(model_id, f"{origin} (source {source_repo}@{source_revision})")
    return bundle
