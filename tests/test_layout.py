"""Every tracked file sits where AGENTS.md says it belongs."""

import subprocess
from fnmatch import fnmatch
from pathlib import Path

import pytest

from vtb.roster import SLUGS

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {
    ".git-blame-ignore-revs",
    ".github",
    ".gitignore",
    "AGENTS.md",
    "CITATION.cff",
    "CLAUDE.md",
    "CONTEXT.md",
    "LICENSE",
    "README.md",
    "docs",
    "hf",
    "pyproject.toml",
    "scripts",
    "tests",
    "uv.lock",
    "vtb",
}
MARKDOWN = (
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "CONTEXT.md",
    "docs/*.md",
    "hf/*/*.md",
    ".github/*.md",
    ".github/ISSUE_TEMPLATE/*.md",
)


@pytest.fixture(scope="module")
def tracked() -> list[str]:
    return subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()


def test_top_level(tracked):
    stray = sorted({path.split("/")[0] for path in tracked} - TOP_LEVEL)
    assert not stray, (
        f"{stray} sit outside the AGENTS.md layout; notes and one-off work go in tmp/<task>/"
    )


def test_markdown(tracked):
    stray = [
        path
        for path in tracked
        if path.endswith(".md")
        and not any(fnmatch(path, pattern) for pattern in MARKDOWN)
    ]
    assert not stray, (
        f"{stray}: notes and checklists go in tmp/<task>/, findings in the PR body"
    )


def test_no_per_tower_test_files(tracked):
    stray = [
        path
        for path in tracked
        if fnmatch(path, "tests/test_*.py")
        and any(slug in Path(path).stem for slug in SLUGS)
    ]
    assert not stray, f"{stray}: per-Tower cases go in tests/test_towers.py"
