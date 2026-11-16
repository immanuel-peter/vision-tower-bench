"""Every published Tower against the checkpoint it came from. See vtb/parity/."""

import os

import pytest

from vtb import parity

DEVICE = os.environ.get("VTB_PARITY_DEVICE", "cpu")

pytestmark = pytest.mark.skipif(os.environ.get("VTB_SKIP_WEIGHTS") == "1", reason="VTB_SKIP_WEIGHTS=1")


@pytest.mark.parametrize("slug", parity.released())
def test_release_matches_its_pinned_parent(slug):
    if parity.requires_cuda(slug) and not DEVICE.startswith("cuda"):
        pytest.skip(f"{slug} parity needs VTB_PARITY_DEVICE=cuda")
    report = parity.validate(slug, DEVICE)
    failing = {key: cell for section in ("weights", "forward") for key, cell in report.get(section, {}).items()
               if not cell.get("bit_exact", cell.get("passed", True))}
    assert report["passed"], failing
