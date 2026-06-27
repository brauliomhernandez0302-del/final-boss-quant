"""Leakage-safe contact-adjusted Team Defense PIT snapshots.

Metric definition
-----------------
``contact_adjusted_defense_proxy`` is the mean, over eligible balls in play
with Statcast xBA, of::

    actual_out_on_batter - (1 - xBA)

Positive values mean the defense converted more batter-outs than expected for
the observed launch speed/angle contact quality. This avoids rewarding a team
merely because its pitchers allowed weak contact. It is not official OAA or
DRS: xBA lacks the complete fielder positioning, route, park geometry and
opportunity model required by those metrics.

Event contract
--------------
* Fielding team: home team in Top innings; away team in Bot/Bottom innings.
* Included BIP outcomes are enumerated in ``BIP_EVENTS``.
* Strikeouts, walks, intentional walks, HBP, catcher interference and every
  other non-BIP event are excluded.
* Home runs are explicitly excluded because the initial proxy models plays a
  defense can plausibly convert in the field.
* BIP without xBA are counted but do not contribute a fabricated value.
* One contribution maximum per ``(game_pk, at_bat_number)``.

This module is isolated from live and default backtest paths.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .pit_cache import PITCache, PITCacheRecord
from .raw_savant_events_cache import RawSavantEventsCache, RawSavantTeamDefenseEvent


METRIC_VERSION = "contact_adjusted_defense_proxy_v1"
NON_MLB_TEAM_IDS = frozenset({"AL", "NL"})

# Batter is recorded out on the contacted ball.
BATTER_OUT_EVENTS = frozenset(
    {
        "field_out",
        "grounded_into_double_play",
        "double_play",
        "triple_play",
        "fielders_choice_out",
        "sac_fly",
        "sac_bunt",
        "sac_fly_double_play",
    }
)

# Batter reaches safely or is not retired, despite a fieldable contacted ball.
BATTER_SAFE_EVENTS = frozenset(
    {
        "single",
        "double",
        "triple",
        "field_error",
        "fielders_choice",
        "force_out",
    }
)
BIP_EVENTS = BATTER_OUT_EVENTS | BATTER_SAFE_EVENTS
EXPLICIT_NON_BIP_EVENTS = frozenset(
    {
        "strikeout",
        "strikeout_double_play",
        "walk",
        "intent_walk",
        "hit_by_pitch",
        "catcher_interf",
        "home_run",
        "truncated_pa",
    }
)


class TeamDefensePITNamespaces:
    TEAM_DEFENSE_ROLLING = "savant.team_defense.rolling"
    TEAM_DEFENSE_PRIOR_BASELINE = "savant.team_defense.prior_baseline"


class TeamDefensePITSources:
    TEAM_DEFENSE_ROLLING = "baseball_savant_team_defense_rolling"
    TEAM_DEFENSE_PRIOR_BASELINE = "baseball_savant_team_defense_prior_baseline"


@dataclass(frozen=True)
class TeamDefenseMetrics:
    team_id: str
    plate_appearances_seen: int
    bip_count: int
    xba_bip_count: int
    missing_xba_bip_count: int
    outs_on_bip: int
    expected_outs: float | None
    contact_adjusted_outs: float | None
    contact_adjusted_defense_proxy: float | None
    sample_size_status: str


@dataclass(frozen=True)
class TeamDefenseAggregationResult:
    rows: dict[str, TeamDefenseMetrics]
    rows_processed: int
    duplicate_pa_rows: int
    invalid_fielding_team_rows: int
    excluded_home_runs: int
    excluded_non_bip_events: int
    excluded_non_mlb_teams: tuple[str, ...]


class _Accumulator:
    def __init__(self) -> None:
        self.plate_appearances_seen = 0
        self.bip_count = 0
        self.xba_bip_count = 0
        self.missing_xba_bip_count = 0
        self.outs_on_bip = 0
        self.expected_outs = 0.0
        self.contact_adjusted_outs = 0.0

    def add(self, event: RawSavantTeamDefenseEvent) -> None:
        self.plate_appearances_seen += 1
        if event.events not in BIP_EVENTS:
            return
        self.bip_count += 1
        actual_out = 1 if event.events in BATTER_OUT_EVENTS else 0
        self.outs_on_bip += actual_out
        xba = event.estimated_ba_using_speedangle
        if xba is None or not 0.0 <= xba <= 1.0:
            self.missing_xba_bip_count += 1
            return
        expected_out = 1.0 - xba
        self.xba_bip_count += 1
        self.expected_outs += expected_out
        self.contact_adjusted_outs += actual_out - expected_out

    def to_metrics(self, team_id: str) -> TeamDefenseMetrics:
        count = self.xba_bip_count
        return TeamDefenseMetrics(
            team_id=team_id,
            plate_appearances_seen=self.plate_appearances_seen,
            bip_count=self.bip_count,
            xba_bip_count=count,
            missing_xba_bip_count=self.missing_xba_bip_count,
            outs_on_bip=self.outs_on_bip,
            expected_outs=round(self.expected_outs, 6) if count else None,
            contact_adjusted_outs=(
                round(self.contact_adjusted_outs, 6) if count else None
            ),
            contact_adjusted_defense_proxy=(
                round(self.contact_adjusted_outs / count, 6) if count else None
            ),
            sample_size_status=_sample_size_status(count),
        )


class TeamDefensePITBuilder:
    """Aggregate and persist rolling team defense snapshots from raw Savant."""

    NAMESPACE = TeamDefensePITNamespaces.TEAM_DEFENSE_ROLLING
    SOURCE = TeamDefensePITSources.TEAM_DEFENSE_ROLLING
    METRIC_VERSION = METRIC_VERSION

    def __init__(
        self,
        *,
        raw_cache_db: Path | str,
        pit_cache_db: Path | str | None = None,
        pit_cache: PITCache | None = None,
        raw_cache: RawSavantEventsCache | None = None,
    ):
        if pit_cache_db is None and pit_cache is None:
            raise ValueError("pit_cache_db is required unless pit_cache is provided")
        self.raw_cache = raw_cache or RawSavantEventsCache(raw_cache_db)
        self.pit_cache = pit_cache or PITCache(pit_cache_db)  # type: ignore[arg-type]

    def aggregate_window(self, *, start_date: str, end_date: str) -> TeamDefenseAggregationResult:
        accumulators: dict[str, _Accumulator] = {}
        seen_pa: set[tuple[int, int]] = set()
        rows_processed = 0
        duplicate_pa_rows = 0
        invalid_fielding_team_rows = 0
        excluded_home_runs = 0
        excluded_non_bip_events = 0
        excluded_non_mlb_teams: set[str] = set()

        for event in self.raw_cache.iter_team_defense_events_by_date_range(
            start_date=start_date,
            end_date=end_date,
        ):
            rows_processed += 1
            pa_key = (event.game_pk, event.at_bat_number)
            if pa_key in seen_pa:
                duplicate_pa_rows += 1
                continue
            seen_pa.add(pa_key)

            fielding_team = fielding_team_for_event(event)
            if fielding_team is None:
                invalid_fielding_team_rows += 1
                continue
            if fielding_team in NON_MLB_TEAM_IDS:
                excluded_non_mlb_teams.add(fielding_team)
                continue

            if event.events == "home_run":
                excluded_home_runs += 1
            elif event.events not in BIP_EVENTS:
                excluded_non_bip_events += 1

            accumulator = accumulators.setdefault(fielding_team, _Accumulator())
            accumulator.add(event)

        return TeamDefenseAggregationResult(
            rows={
                team_id: accumulator.to_metrics(team_id)
                for team_id, accumulator in sorted(accumulators.items())
            },
            rows_processed=rows_processed,
            duplicate_pa_rows=duplicate_pa_rows,
            invalid_fielding_team_rows=invalid_fielding_team_rows,
            excluded_home_runs=excluded_home_runs,
            excluded_non_bip_events=excluded_non_bip_events,
            excluded_non_mlb_teams=tuple(sorted(excluded_non_mlb_teams)),
        )

    def persist_cutoff(
        self,
        *,
        season: int,
        season_start_date: str,
        as_of_date: str,
        window_days: int = 60,
        fetched_at: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Persist a trailing-window snapshot ending on ``as_of_date``."""
        if window_days < 1:
            raise ValueError("window_days must be positive")
        cutoff_day = date.fromisoformat(as_of_date)
        season_start = date.fromisoformat(season_start_date)
        window_start = max(season_start, cutoff_day - timedelta(days=window_days - 1))
        source_start = window_start.isoformat()
        result = self.aggregate_window(start_date=source_start, end_date=as_of_date)
        fingerprint = _source_fingerprint(
            source=self.SOURCE,
            season=season,
            start_date=source_start,
            end_date=as_of_date,
            rows_processed=result.rows_processed,
            teams=len(result.rows),
            metric_version=self.METRIC_VERSION,
            input_fingerprints=self.raw_cache.source_fingerprints_by_date_range(
                start_date=source_start,
                end_date=as_of_date,
            ),
        )
        cutoff = f"{as_of_date}T23:59:59Z"
        fetched = fetched_at or datetime.now(timezone.utc).isoformat()
        persisted: dict[str, dict[str, Any]] = {}
        for team_id, metrics in result.rows.items():
            payload = _metrics_payload(
                metrics,
                source_window_start_date=source_start,
                source_window_end_date=as_of_date,
                source_fingerprint=fingerprint,
            )
            payload.update(
                {
                    "season": int(season),
                    "duplicate_pa_rows": result.duplicate_pa_rows,
                    "invalid_fielding_team_rows": result.invalid_fielding_team_rows,
                    "excluded_home_runs": result.excluded_home_runs,
                    "excluded_non_bip_events": result.excluded_non_bip_events,
                }
            )
            self.pit_cache.save_record(
                namespace=self.NAMESPACE,
                entity_id=team_id,
                season=season,
                as_of_date=cutoff,
                source=self.SOURCE,
                source_fingerprint=fingerprint,
                data=payload,
                fetched_at=fetched,
            )
            persisted[team_id] = payload
        return persisted

    def get_latest_snapshot(
        self,
        *,
        team_id: int | str,
        season: int,
        requested_as_of_date: str,
    ) -> PITCacheRecord | None:
        return self.pit_cache.get_latest(
            namespace=self.NAMESPACE,
            entity_id=team_id,
            season=season,
            as_of_date=requested_as_of_date,
            source=self.SOURCE,
        )


class TeamDefenseDailySnapshotBuilder:
    """Read current and prior defense records at or before a requested cutoff."""

    SNAPSHOT_VERSION = "team_defense_daily_snapshot_v1"

    def __init__(self, pit_cache: PITCache | None = None, *, cache_db: Path | str | None = None):
        if pit_cache is None and cache_db is None:
            raise ValueError("pit_cache or cache_db is required")
        self.cache = pit_cache or PITCache(cache_db)  # type: ignore[arg-type]

    def build_for_game(self, *, team_id: int | str, season: int, game_date: str) -> dict[str, Any]:
        game_day = datetime.strptime(game_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        requested = (game_day - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self.build_snapshot(
            team_id=team_id,
            season=season,
            requested_as_of_date=requested,
        )

    def build_snapshot(
        self,
        *,
        team_id: int | str,
        season: int,
        requested_as_of_date: str,
    ) -> dict[str, Any]:
        current = self.cache.get_latest(
            namespace=TeamDefensePITNamespaces.TEAM_DEFENSE_ROLLING,
            entity_id=team_id,
            season=season,
            as_of_date=requested_as_of_date,
            source=TeamDefensePITSources.TEAM_DEFENSE_ROLLING,
        )
        prior = self.cache.get_latest(
            namespace=TeamDefensePITNamespaces.TEAM_DEFENSE_PRIOR_BASELINE,
            entity_id=team_id,
            season=season,
            as_of_date=requested_as_of_date,
            source=TeamDefensePITSources.TEAM_DEFENSE_PRIOR_BASELINE,
        )
        current_data = current.data if current else {}
        prior_data = prior.data if prior else {}
        return {
            "found": bool(current or prior),
            "team_id": _coerce_int(team_id),
            "season": int(season),
            "requested_as_of_date": requested_as_of_date,
            "current_defense_found": current is not None,
            "prior_baseline_found": prior is not None,
            "current_as_of_date": current.as_of_date if current else None,
            "prior_baseline_as_of_date": prior.as_of_date if prior else None,
            "contact_adjusted_defense_proxy": current_data.get(
                "contact_adjusted_defense_proxy"
            ),
            "bip_count": current_data.get("bip_count"),
            "xba_bip_count": current_data.get("xba_bip_count"),
            "sample_size_status": current_data.get("sample_size_status", "missing"),
            "source_window_start_date": current_data.get("source_window_start_date"),
            "source_window_end_date": current_data.get("source_window_end_date"),
            "prior_contact_adjusted_defense_proxy": prior_data.get(
                "contact_adjusted_defense_proxy"
            ),
            "prior_bip_count": prior_data.get("bip_count"),
            "prior_xba_bip_count": prior_data.get("xba_bip_count"),
            "prior_source_window_start_date": prior_data.get("source_window_start_date"),
            "prior_source_window_end_date": prior_data.get("source_window_end_date"),
            "source_fingerprints": {
                "current_defense_pit": current.source_fingerprint if current else None,
                "prior_season_defense_baseline": prior.source_fingerprint if prior else None,
            },
            "snapshot_version": self.SNAPSHOT_VERSION,
        }


def fielding_team_for_event(event: RawSavantTeamDefenseEvent) -> str | None:
    """Return fielding team after validating inning-side attribution."""
    home = (event.home_team or "").strip()
    away = (event.away_team or "").strip()
    half = (event.inning_topbot or "").strip().lower()
    if not home or not away or home == away:
        return None
    if half in {"top", "t"}:
        fielding, expected_batting = home, away
    elif half in {"bot", "bottom", "b"}:
        fielding, expected_batting = away, home
    else:
        return None
    if event.batting_team and event.batting_team.strip() != expected_batting:
        return None
    return fielding


def _metrics_payload(
    metrics: TeamDefenseMetrics,
    *,
    source_window_start_date: str,
    source_window_end_date: str,
    source_fingerprint: str,
) -> dict[str, Any]:
    return {
        "team_id": metrics.team_id,
        "metric_name": "contact_adjusted_defense_proxy",
        "metric_version": METRIC_VERSION,
        "contact_adjusted_defense_proxy": metrics.contact_adjusted_defense_proxy,
        "plate_appearances_seen": metrics.plate_appearances_seen,
        "bip_count": metrics.bip_count,
        "xba_bip_count": metrics.xba_bip_count,
        "missing_xba_bip_count": metrics.missing_xba_bip_count,
        "outs_on_bip": metrics.outs_on_bip,
        "expected_outs": metrics.expected_outs,
        "contact_adjusted_outs": metrics.contact_adjusted_outs,
        "sample_size_status": metrics.sample_size_status,
        "source_window_start_date": source_window_start_date,
        "source_window_end_date": source_window_end_date,
        "source_fingerprint": source_fingerprint,
        "event_contract": {
            "batter_out_events": sorted(BATTER_OUT_EVENTS),
            "batter_safe_events": sorted(BATTER_SAFE_EVENTS),
            "home_run_treatment": "excluded_not_plausibly_fieldable",
            "non_bip_treatment": "excluded",
            "missing_xba_treatment": "counted_bip_no_metric_contribution",
        },
        "limitations": "contact-adjusted team proxy; not official OAA or DRS",
    }


def _source_fingerprint(
    *,
    source: str,
    season: int,
    start_date: str,
    end_date: str,
    rows_processed: int,
    teams: int,
    metric_version: str,
    input_fingerprints: tuple[str, ...] = (),
) -> str:
    payload = {
        "source": source,
        "season": int(season),
        "start_date": start_date,
        "end_date": end_date,
        "rows_processed": int(rows_processed),
        "teams": int(teams),
        "metric_version": metric_version,
        "input_fingerprints": input_fingerprints,
        "event_contract": {
            "out": sorted(BATTER_OUT_EVENTS),
            "safe": sorted(BATTER_SAFE_EVENTS),
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return (
        f"savant:raw:team_defense:{metric_version}:{season}:"
        f"{start_date}:{end_date}:{rows_processed}:{teams}:{digest}"
    )


def _sample_size_status(xba_bip_count: int) -> str:
    if xba_bip_count <= 0:
        return "missing"
    if xba_bip_count < 100:
        return "thin"
    return "ok"


def _coerce_int(value: int | str) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value
