import vtb
from vtb.adapters.dinov2 import DINOv2Adapter
from vtb.extract import ADAPTERS
from vtb.roster import BY_SLUG, ROSTER


def test_every_slug_is_unique_and_registered_for_extraction():
    assert len(BY_SLUG) == len(ROSTER)
    assert set(ADAPTERS) == set(BY_SLUG)


def test_roster_model_ids_match_their_adapters():
    for tower in ROSTER:
        adapter = tower.adapter()
        checkpoint = getattr(adapter, "checkpoint", None)
        assert tower.model_id == (
            checkpoint.model_id if checkpoint else adapter.model_id
        ), tower.slug


def test_public_api_exposes_the_roster():
    assert vtb.SLUGS == tuple(tower.slug for tower in vtb.ROSTER)
    assert vtb.BY_SLUG["dinov2"].adapter() is DINOv2Adapter
    assert set(vtb.__all__) <= set(dir(vtb))
