from vtb.extract import ADAPTERS
from vtb.roster import BY_SLUG, ROSTER


def test_every_slug_is_unique_and_registered_for_extraction():
    assert len(BY_SLUG) == len(ROSTER)
    assert set(ADAPTERS) == set(BY_SLUG)


def test_roster_model_ids_match_their_adapters():
    for tower in ROSTER:
        adapter = tower.adapter()
        checkpoint = getattr(adapter, "checkpoint", None)
        assert tower.model_id == (checkpoint.model_id if checkpoint else adapter.model_id), tower.slug
