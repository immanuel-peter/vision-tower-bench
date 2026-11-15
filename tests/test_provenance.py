import json
import logging

import pytest
import torch

from vtb import provenance
from vtb.adapters import deepseek_v41, gemma4, glm5, mimo26, minimax_m3, nemotron35, nemotron_omni
from vtb.provenance import check_provenance, open_release


class Fetched(Exception):
    pass


@pytest.fixture
def hub(monkeypatch):
    calls = []

    def snapshot_download(repo, revision):
        calls.append((repo, revision))
        raise Fetched

    monkeypatch.setattr(provenance, "snapshot_download", snapshot_download)
    for name in ("MIMO26_PRO", "MINIMAX_M3", "NEMOTRON35_SUPER"):
        monkeypatch.delenv(f"VTB_{name}_BUNDLE", raising=False)
    return calls


def write(path, name, data):
    (path / name).write_text(json.dumps(data))


def test_manifest_and_config_records_are_both_checked(tmp_path):
    write(tmp_path, "config.json", {"source": "org/parent", "revision": "a"})
    check_provenance(tmp_path, "org/parent", "a", required=True)
    with pytest.raises(ValueError, match="org/parent@a does not match org/parent@b"):
        check_provenance(tmp_path, "org/parent", "b")
    write(tmp_path, "export_manifest.json", {"source_repo": "org/parent", "source_revision": "b"})
    check_provenance(tmp_path, "org/parent", "b")


def test_missing_record_fails_only_when_required(tmp_path):
    check_provenance(tmp_path, "org/parent", "a")
    with pytest.raises(ValueError, match="records no source provenance"):
        check_provenance(tmp_path, "org/parent", "a", required=True)


def test_open_release_logs_repo_at_revision(monkeypatch, tmp_path, caplog):
    write(tmp_path, "export_manifest.json", {"source_repo": "org/parent", "source_revision": "p"})
    monkeypatch.setattr(provenance, "snapshot_download", lambda repo, revision: str(tmp_path))
    with caplog.at_level(logging.INFO, logger="vtb.provenance"):
        assert open_release("me/tower", "r", "org/parent", "p") == tmp_path
    assert "me/tower loaded weights from me/tower@r (source org/parent@p)" in caplog.text


@pytest.mark.parametrize("load,release", [
    (lambda: gemma4.load_parts(torch.bfloat16), (gemma4.MODEL_ID, gemma4.REVISION)),
    (lambda: glm5.load_tower(torch.bfloat16), (glm5.MODEL_ID, glm5.REVISION)),
    (lambda: deepseek_v41.load_parts(torch.bfloat16), (deepseek_v41.MODEL_ID, deepseek_v41.REVISION)),
    (lambda: nemotron_omni.load_parts(torch.bfloat16), (nemotron_omni.MODEL_ID, nemotron_omni.REVISION)),
    (lambda: minimax_m3.load_parts(torch.bfloat16, minimax_m3.FLASH), (minimax_m3.FLASH.model_id, minimax_m3.FLASH.release_revision)),
    (lambda: mimo26.load_model(mimo26.PRO, torch.bfloat16), (mimo26.PRO.model_id, mimo26.PRO.release_revision)),
    (lambda: nemotron35.load_parts(nemotron35.SUPER, torch.bfloat16), (nemotron35.SUPER.model_id, nemotron35.SUPER.release_revision)),
], ids=["gemma4", "glm5", "deepseek_v41", "nemotron_omni", "minimax_m3", "mimo26", "nemotron35"])
def test_default_load_fetches_the_pinned_release(hub, load, release):
    with pytest.raises(Fetched):
        load()
    assert hub == [release]


@pytest.mark.parametrize("checkpoint", [mimo26.PRO, nemotron35.SUPER, minimax_m3.FLASH], ids=lambda c: c.name)
def test_bundle_override_replaces_the_hub_and_is_provenance_checked(hub, monkeypatch, tmp_path, checkpoint):
    write(tmp_path, "export_manifest.json", {"source_repo": checkpoint.repo, "source_revision": "0" * 40})
    monkeypatch.setenv(f"VTB_{checkpoint.name.upper()}_BUNDLE", str(tmp_path))
    with pytest.raises(ValueError, match="provenance"):
        checkpoint.bundle
    assert hub == []
