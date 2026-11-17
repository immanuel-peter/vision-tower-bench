from vtb.adapters.base import TowerAdapter
from vtb.feature_batch import FeatureBatch
from vtb.roster import BY_SLUG, ROSTER, SLUGS, Tower


def load(slug: str, **kwargs) -> TowerAdapter:
    return BY_SLUG[slug].adapter()(**kwargs)


__all__ = [
    "BY_SLUG",
    "ROSTER",
    "SLUGS",
    "FeatureBatch",
    "Tower",
    "TowerAdapter",
    "load",
]
