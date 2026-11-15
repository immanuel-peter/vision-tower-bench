import ast
import inspect
import re
from pathlib import Path

import pytest

from vtb import provenance
from vtb.adapters import deepseek_v41, dinov2, gemma4, glm5, kimi_k26, mimo26, minimax_m3, moonvit_v2, muse_glimmer, nemotron35, nemotron_omni, qwen3_5, siglip2

MODULES = (dinov2, siglip2, qwen3_5, muse_glimmer, kimi_k26, moonvit_v2, gemma4, glm5, minimax_m3, nemotron_omni, mimo26, nemotron35, provenance)
SHA = re.compile(r"[0-9a-f]{40}")


def hub_calls(module):
    tree = ast.parse(Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in {"from_pretrained", "hf_hub_download", "snapshot_download"}:
                yield node


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_every_hub_load_passes_a_revision(module):
    calls = list(hub_calls(module))
    assert calls
    for call in calls:
        keywords = {k.arg: k.value for k in call.keywords}
        local = "local_files_only" in keywords and ast.literal_eval(keywords["local_files_only"]) is True
        assert "revision" in keywords or local, f"{module.__name__}:{call.lineno} loads from the Hub without a revision"


@pytest.mark.parametrize("module", (dinov2, siglip2, muse_glimmer, kimi_k26, gemma4, glm5, nemotron_omni, deepseek_v41), ids=lambda m: m.__name__)
def test_module_revision_is_a_commit_sha(module):
    assert SHA.fullmatch(module.REVISION)


def test_checkpoint_revisions_are_commit_shas():
    for checkpoint in qwen3_5.CHECKPOINTS.values():
        assert SHA.fullmatch(checkpoint.revision) and SHA.fullmatch(checkpoint.release_revision)
    for checkpoint in moonvit_v2.CHECKPOINTS.values():
        assert SHA.fullmatch(checkpoint.revision) and SHA.fullmatch(checkpoint.bundle_revision)
    assert inspect.signature(moonvit_v2.load_projector_weights).parameters["revision"].default == moonvit_v2.K3.bundle_revision
    for module in (minimax_m3, mimo26, nemotron35):
        for checkpoint in module.CHECKPOINTS.values():
            assert SHA.fullmatch(checkpoint.revision) and SHA.fullmatch(checkpoint.release_revision)


def test_dinov2_matches_results_readme():
    readme = (Path(__file__).parents[1] / "results" / "README.md").read_text()
    assert f"`{dinov2.MODEL_ID}` at revision\n`{dinov2.REVISION}`" in readme
