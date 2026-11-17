"""Parents and upstream Towers load at a pinned commit; our own releases load their latest revision."""

import ast
import re
import sys
from pathlib import Path

import pytest

from vtb import parity, provenance
from vtb.roster import BY_SLUG, ROSTER

MODULES = (
    *dict.fromkeys(sys.modules[tower.adapter().__module__] for tower in ROSTER),
    provenance,
)
OURS = {
    sys.modules[BY_SLUG[slug].adapter().__module__] for slug in parity.released()
} | {provenance}
SHA = re.compile(r"[0-9a-f]{40}")


def hub_calls(module):
    tree = ast.parse(Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if name in {"from_pretrained", "hf_hub_download", "snapshot_download"}:
                yield node


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_every_parent_and_upstream_hub_load_passes_a_revision(module):
    calls = list(hub_calls(module))
    assert calls
    for call in calls:
        keywords = {k.arg: k.value for k in call.keywords}
        local = (
            "local_files_only" in keywords
            and ast.literal_eval(keywords["local_files_only"]) is True
        )
        repo = ast.unparse(call.args[0]) if call.args else ""
        own_release = module in OURS and "model_id" in repo.lower()
        assert "revision" in keywords or local or own_release, (
            f"{module.__name__}:{call.lineno} loads {repo} from the Hub without a revision"
        )


@pytest.mark.parametrize("module", MODULES[:-1], ids=lambda m: m.__name__)
def test_pinned_revisions_are_commit_shas(module):
    pins = [
        value
        for name, value in vars(module).items()
        if name.endswith("REVISION") and isinstance(value, str)
    ]
    pins += [
        checkpoint.revision
        for checkpoint in getattr(module, "CHECKPOINTS", {}).values()
    ]
    assert pins and all(SHA.fullmatch(pin) for pin in pins)
