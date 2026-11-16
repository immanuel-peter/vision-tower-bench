import vtb
from vtb.adapters.dinov2 import DINOv2Adapter


def test_public_api_exposes_the_roster():
    assert vtb.SLUGS == tuple(tower.slug for tower in vtb.ROSTER)
    assert vtb.BY_SLUG["dinov2"].adapter() is DINOv2Adapter
    assert set(vtb.__all__) <= set(dir(vtb))
