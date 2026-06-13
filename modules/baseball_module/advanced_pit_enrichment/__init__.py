"""Advanced point-in-time enrichment foundations for MLB backtesting.

This package is intentionally separate from the live FanGraphs and Savant
fetchers. It should remain disconnected from the live pipeline until the PIT
fetchers and snapshot builders are fully validated.
"""

from .advanced_pitcher_snapshot_builder import AdvancedPitcherSnapshotBuilder
from .fangraphs_daily_pit_persistence import FanGraphsDailyPITPersistence
from .pit_cache import PITCache, PITCacheRecord
from .raw_savant_events_cache import RawSavantEvent, RawSavantEventsCache
from .savant_daily_aggregator import SavantDailyAggregator, SavantDailyPitcherMetrics
from .savant_raw_ingestor import SavantRawIngestionSummary, SavantRawIngestor
from .savant_rolling_pit_builder import SavantRollingPitcherMetrics, SavantRollingPITBuilder
from .savant_rolling_pit_persistence import SavantRollingPITPersistence

__all__ = [
    "AdvancedPitcherSnapshotBuilder",
    "FanGraphsDailyPITPersistence",
    "PITCache",
    "PITCacheRecord",
    "RawSavantEvent",
    "RawSavantEventsCache",
    "SavantDailyAggregator",
    "SavantDailyPitcherMetrics",
    "SavantRawIngestionSummary",
    "SavantRawIngestor",
    "SavantRollingPitcherMetrics",
    "SavantRollingPITBuilder",
    "SavantRollingPITPersistence",
]
