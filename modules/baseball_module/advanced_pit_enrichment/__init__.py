"""Advanced point-in-time enrichment foundations for MLB backtesting.

This package is intentionally separate from the live FanGraphs and Savant
fetchers. It should remain disconnected from the live pipeline until the PIT
fetchers and snapshot builders are fully validated.
"""

from .advanced_pitcher_snapshot_builder import AdvancedPitcherSnapshotBuilder
from .pit_cache import PITCache, PITCacheRecord
from .raw_savant_events_cache import RawSavantEvent, RawSavantEventsCache
from .savant_daily_aggregator import SavantDailyAggregator, SavantDailyPitcherMetrics
from .savant_raw_ingestor import SavantRawIngestionSummary, SavantRawIngestor

__all__ = [
    "AdvancedPitcherSnapshotBuilder",
    "PITCache",
    "PITCacheRecord",
    "RawSavantEvent",
    "RawSavantEventsCache",
    "SavantDailyAggregator",
    "SavantDailyPitcherMetrics",
    "SavantRawIngestionSummary",
    "SavantRawIngestor",
]
