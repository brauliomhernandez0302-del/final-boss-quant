"""Advanced point-in-time enrichment foundations for MLB backtesting.

This package is intentionally separate from the live FanGraphs and Savant
fetchers. It should remain disconnected from the live pipeline until the PIT
fetchers and snapshot builders are fully validated.
"""

from .advanced_pitcher_snapshot_builder import AdvancedPitcherSnapshotBuilder
from .advanced_pitcher_daily_snapshot_builder import AdvancedPitcherDailySnapshotBuilder
from .fangraphs_daily_pit_persistence import FanGraphsDailyPITPersistence
from .pit_cache import PITCache, PITCacheRecord
from .pitcher_prior_baseline import PitcherPriorBaselinePersistence
from .pitcher_engine_snapshot_adapter import adapt_unified_pitcher_snapshot
from .raw_savant_events_cache import (
    RawSavantEvent,
    RawSavantEventsCache,
    RawSavantTeamDefenseEvent,
)
from .savant_daily_aggregator import SavantDailyAggregator, SavantDailyPitcherMetrics
from .savant_raw_ingestor import SavantRawIngestionSummary, SavantRawIngestor
from .savant_offense_daily_aggregator import (
    SavantBatterRollingPITPersistence,
    SavantDailyBatterOffenseMetrics,
    SavantDailyTeamOffenseMetrics,
    SavantOffenseDailyAggregator,
    SavantOffenseRollingBuilder,
    SavantRollingBatterOffenseMetrics,
    SavantRollingTeamOffenseMetrics,
    SavantRollingTeamOffenseResult,
    SavantTeamOffenseDailyResult,
    SavantTeamOffenseRollingPITPersistence,
)
from .savant_rolling_pit_builder import SavantRollingPitcherMetrics, SavantRollingPITBuilder
from .savant_rolling_pit_persistence import SavantRollingPITPersistence
from .tte_daily_snapshot_builder import (
    TTEDailySnapshotBuilder,
    TTEPITNamespaces,
    TTEPITSources,
    previous_day_cutoff_for_game_date,
)
from .tte_pit_adapter import adapt_tte_pit_snapshot_to_lambda
from .tte_prior_baseline_builder import TTEPriorBaselineBuilder, TTEPriorBaselineBuildResult
from .team_defense_pit_builder import (
    TeamDefenseAggregationResult,
    TeamDefenseDailySnapshotBuilder,
    TeamDefenseMetrics,
    TeamDefensePITBuilder,
    TeamDefensePITNamespaces,
    TeamDefensePITSources,
    fielding_team_for_event,
)
from .team_defense_prior_baseline import (
    TeamDefensePriorBaseline,
    TeamDefensePriorBaselineBuildResult,
)
from .defense_pit_adapter import adapt_defense_pit_snapshot

__all__ = [
    "AdvancedPitcherSnapshotBuilder",
    "AdvancedPitcherDailySnapshotBuilder",
    "FanGraphsDailyPITPersistence",
    "PITCache",
    "PITCacheRecord",
    "PitcherPriorBaselinePersistence",
    "adapt_unified_pitcher_snapshot",
    "RawSavantEvent",
    "RawSavantEventsCache",
    "RawSavantTeamDefenseEvent",
    "SavantDailyAggregator",
    "SavantDailyPitcherMetrics",
    "SavantRawIngestionSummary",
    "SavantRawIngestor",
    "SavantBatterRollingPITPersistence",
    "SavantDailyBatterOffenseMetrics",
    "SavantDailyTeamOffenseMetrics",
    "SavantOffenseDailyAggregator",
    "SavantOffenseRollingBuilder",
    "SavantRollingBatterOffenseMetrics",
    "SavantRollingTeamOffenseMetrics",
    "SavantRollingTeamOffenseResult",
    "SavantTeamOffenseDailyResult",
    "SavantTeamOffenseRollingPITPersistence",
    "SavantRollingPitcherMetrics",
    "SavantRollingPITBuilder",
    "SavantRollingPITPersistence",
    "TTEDailySnapshotBuilder",
    "TTEPITNamespaces",
    "TTEPITSources",
    "adapt_tte_pit_snapshot_to_lambda",
    "TTEPriorBaselineBuilder",
    "TTEPriorBaselineBuildResult",
    "TeamDefenseAggregationResult",
    "TeamDefenseDailySnapshotBuilder",
    "TeamDefenseMetrics",
    "TeamDefensePITBuilder",
    "TeamDefensePITNamespaces",
    "TeamDefensePITSources",
    "TeamDefensePriorBaseline",
    "TeamDefensePriorBaselineBuildResult",
    "adapt_defense_pit_snapshot",
    "fielding_team_for_event",
    "previous_day_cutoff_for_game_date",
]
