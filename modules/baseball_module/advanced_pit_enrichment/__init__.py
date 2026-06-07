"""Advanced point-in-time enrichment foundations for MLB backtesting.

This package is intentionally separate from the live FanGraphs and Savant
fetchers. It should remain disconnected from the live pipeline until the PIT
fetchers and snapshot builders are fully validated.
"""

from .pit_cache import PITCache, PITCacheRecord

__all__ = ["PITCache", "PITCacheRecord"]
