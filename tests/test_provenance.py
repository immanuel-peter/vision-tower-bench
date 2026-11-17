import json
import logging

import pytest
import torch

from vtb import provenance
from vtb.adapters import (
    deepseek_v41,
    gemma4,
    glm5,
    mimo26,
    minimax_m3,
    nemotron35,
    nemotron_omni,
)
from vtb.provenance import check_provenance, open_release


class Fetched(Exception):
    pass


@pytest.fixture
def hub(monkeypatch):
    calls = []

    def snapshot_download(repo):
        calls.append(repo)
        raise Fetched

    monkeypatch.setattr(provenance, "snapshot_download", snapshot_download)
    for name in ("MIMO26_PRO", "MINIMAX_M3", "NEMOTRON35_SUPER"):
        monkeypatch.delenv(f"VTB_{name}_BUNDLE", raising=False)
    return calls


def write(path, name, data):
    (path / name).write_text(json.dumps(data))


def test_manifest_and_config_records_are_both_checked(tmp_path):
    write(tmp_path, "config.json", {"source": "org/parent", "revision": "a"})
    check_provenance(tmp_path, "org/parent", "a")
    with pytest.raises(ValueError, match="org/parent@a does not match org/parent@b"):
        check_provenance(tmp_path, "org/parent", "b")
    write(
        tmp_path,
        "export_manifest.json",
        {"source_repo": "org/parent", "source_revision": "b"},
    )
    check_provenance(tmp_path, "org/parent", "b")


def test_bundle_without_a_record_is_accepted(tmp_path):
    check_provenance(tmp_path, "org/parent", "a")


def test_open_release_checks_the_hub_snapshot_against_its_parent(
    monkeypatch, tmp_path, caplog
):
    write(
        tmp_path,
        "export_manifest.json",
        {"source_repo": "org/parent", "source_revision": "p"},
    )
    monkeypatch.setattr(provenance, "snapshot_download", lambda repo: str(tmp_path))
    with caplog.at_level(logging.INFO, logger="vtb.provenance"):
        assert open_release("me/tower", "org/parent", "p") == tmp_path
    assert "me/tower loaded weights from me/tower (source org/parent@p)" in caplog.text
    with pytest.raises(ValueError, match="provenance"):
        open_release("me/tower", "org/parent", "other")


@pytest.mark.parametrize(
    "load,release",
    [
        (lambda: gemma4.load_parts(torch.bfloat16), gemma4.MODEL_ID),
        (lambda: glm5.load_tower(torch.bfloat16), glm5.MODEL_ID),
        (lambda: deepseek_v41.load_parts(torch.bfloat16), deepseek_v41.MODEL_ID),
        (lambda: nemotron_omni.load_parts(torch.bfloat16), nemotron_omni.MODEL_ID),
        (
            lambda: minimax_m3.load_parts(torch.bfloat16, minimax_m3.FLASH),
            minimax_m3.FLASH.model_id,
        ),
        (lambda: mimo26.load_model(mimo26.PRO, torch.bfloat16), mimo26.PRO.model_id),
        (
            lambda: nemotron35.load_parts(nemotron35.SUPER, torch.bfloat16),
            nemotron35.SUPER.model_id,
        ),
    ],
    ids=[
        "gemma4",
        "glm5",
        "deepseek_v41",
        "nemotron_omni",
        "minimax_m3",
        "mimo26",
        "nemotron35",
    ],
)
def test_default_load_fetches_the_latest_release(hub, load, release):
    with pytest.raises(Fetched):
        load()
    assert hub == [release]


@pytest.mark.parametrize(
    "checkpoint", [mimo26.PRO, nemotron35.SUPER, minimax_m3.FLASH], ids=lambda c: c.name
)
def test_bundle_override_replaces_the_hub_and_is_provenance_checked(
    hub, monkeypatch, tmp_path, checkpoint
):
    write(
        tmp_path,
        "export_manifest.json",
        {"source_repo": checkpoint.repo, "source_revision": "0" * 40},
    )
    monkeypatch.setenv(f"VTB_{checkpoint.name.upper()}_BUNDLE", str(tmp_path))
    with pytest.raises(ValueError, match="provenance"):
        checkpoint.bundle
    assert hub == []
