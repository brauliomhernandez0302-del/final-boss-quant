"""Current Team Bullpen PIT snapshots from relief-only Raw Savant facts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .bullpen_relief_appearance_builder import (
    FACT_BUILDER_VERSION,
    ROLE_RULE_VERSION,
    BullpenPITNamespaces,
    BullpenPITSources,
    BullpenReliefAppearanceBuilder,
    ReliefAppearanceFact,
    _fingerprint,
)
from .pit_cache import PITCache
from .raw_savant_events_cache import RawSavantEventsCache


BULLPEN_ROLLING_VERSION = "team_bullpen_rolling_v1"
SAMPLE_BF_THRESHOLD = 200


@dataclass(frozen=True)
class TeamBullpenMetrics:
    team_id: str
    relief_appearances: int
    relief_team_games: int
    relief_pitch_count: int
    relief_batters_faced: int
    strikeouts: int
    walks: int
    k_pct: float | None
    bb_pct: float | None
    k_minus_bb_pct: float | None
    xwoba_against: float | None
    xwoba_count: int
    woba_against: float | None
    woba_numerator: float | None
    woba_denominator: float | None
    barrel_count: int
    contact_count: int
    barrel_per_contact: float | None
    sample_size_status: str


class TeamBullpenPITBuilder:
    """Persist D-1-readable team snapshots without model adjustments."""

    NAMESPACE = BullpenPITNamespaces.TEAM_BULLPEN_ROLLING
    SOURCE = BullpenPITSources.TEAM_BULLPEN_ROLLING
    SNAPSHOT_VERSION = BULLPEN_ROLLING_VERSION

    def __init__(
        self,
        *,
        raw_cache_db: Path | str,
        pit_cache_db: Path | str | None = None,
        pit_cache: PITCache | None = None,
        raw_cache: RawSavantEventsCache | None = None,
        appearance_builder: BullpenReliefAppearanceBuilder | None = None,
    ):
        if pit_cache_db is None and pit_cache is None:
            raise ValueError("pit_cache_db is required unless pit_cache is provided")
        self.pit_cache = pit_cache or PITCache(pit_cache_db)  # type: ignore[arg-type]
        self.raw_cache = raw_cache or RawSavantEventsCache(raw_cache_db)
        self.appearance_builder = appearance_builder or BullpenReliefAppearanceBuilder(
            raw_cache_db=raw_cache_db,
            pit_cache=self.pit_cache,
            raw_cache=self.raw_cache,
        )

    def persist_cutoff(
        self,
        *,
        season: int,
        season_start_date: str,
        as_of_date: str,
        fetched_at: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        cutoff_day = date.fromisoformat(as_of_date)
        start_day = date.fromisoformat(season_start_date)
        if cutoff_day < start_day:
            return {}
        if start_day.year != season or cutoff_day.year != season:
            raise ValueError("current bullpen window must stay inside season")

        result = self.appearance_builder.build(
            start_date=season_start_date,
            end_date=as_of_date,
        )
        grouped = _group_facts(result.facts)
        input_fingerprints = self.raw_cache.source_fingerprints_by_date_range(
            start_date=season_start_date,
            end_date=as_of_date,
        )
        cutoff = f"{as_of_date}T23:59:59Z"
        fetched = fetched_at or datetime.now(timezone.utc).isoformat()
        persisted: dict[str, dict[str, Any]] = {}
        for team_id, facts in sorted(grouped.items()):
            metrics = aggregate_team_bullpen_metrics(team_id, facts)
            workload = workload_facts(facts, cutoff_date=as_of_date)
            team_rejections = [
                item for item in result.rejected_team_games if item.team_id == team_id
            ]
            fingerprint = _fingerprint(
                {
                    "source": self.SOURCE,
                    "snapshot_version": self.SNAPSHOT_VERSION,
                    "role_rule_version": ROLE_RULE_VERSION,
                    "team_id": team_id,
                    "season": season,
                    "source_window": [season_start_date, as_of_date],
                    "fact_fingerprints": sorted(
                        fact.source_fingerprint for fact in facts
                    ),
                    "input_fingerprints": input_fingerprints,
                },
                prefix="savant:raw:bullpen:team-rolling",
            )
            payload = {
                **asdict(metrics),
                **workload,
                "season": int(season),
                "requested_as_of_date": cutoff,
                "source_window_start_date": season_start_date,
                "source_window_end_date": as_of_date,
                "quality_window_start_date": season_start_date,
                "quality_window_end_date": as_of_date,
                "snapshot_version": self.SNAPSHOT_VERSION,
                "fact_builder_version": FACT_BUILDER_VERSION,
                "role_rule_version": ROLE_RULE_VERSION,
                "source_fingerprint": fingerprint,
                "accepted_relief_team_games": metrics.relief_team_games,
                "rejected_relief_team_games": len(team_rejections),
                "rejected_reason_counts": dict(
                    sorted(Counter(item.reason for item in team_rejections).items())
                ),
                "starter_pitches_excluded": result.starter_pitches_excluded,
                "starter_pitches_included": result.starter_pitches_included,
                "duplicate_pitch_contributions_prevented": (
                    result.duplicate_pitch_contributions_prevented
                ),
                "duplicate_pa_contributions_prevented": (
                    result.duplicate_pa_contributions_prevented
                ),
                "input_source_fingerprints": list(input_fingerprints),
                "closer_setup_long_relief_labels": None,
                "roster_fallback_used": False,
                "legacy_bullpen_fallback_used": False,
            }
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


class TeamBullpenDailySnapshotBuilder:
    """Read current and prior bullpen records no later than a requested cutoff."""

    SNAPSHOT_VERSION = "team_bullpen_daily_snapshot_v1"

    def __init__(
        self,
        pit_cache: PITCache | None = None,
        *,
        cache_db: Path | str | None = None,
    ):
        if pit_cache is None and cache_db is None:
            raise ValueError("pit_cache or cache_db is required")
        self.cache = pit_cache or PITCache(cache_db)  # type: ignore[arg-type]

    def build_for_game(
        self,
        *,
        team_id: int | str,
        season: int,
        game_date: str,
    ) -> dict[str, Any]:
        game_day = datetime.strptime(game_date[:10], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        requested = (game_day - timedelta(seconds=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
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
            namespace=BullpenPITNamespaces.TEAM_BULLPEN_ROLLING,
            entity_id=team_id,
            season=season,
            as_of_date=requested_as_of_date,
            source=BullpenPITSources.TEAM_BULLPEN_ROLLING,
        )
        prior = self.cache.get_latest(
            namespace=BullpenPITNamespaces.TEAM_BULLPEN_PRIOR_BASELINE,
            entity_id=team_id,
            season=season,
            as_of_date=requested_as_of_date,
            source=BullpenPITSources.TEAM_BULLPEN_PRIOR_BASELINE,
        )
        return {
            "team_id": str(team_id),
            "season": int(season),
            "requested_as_of_date": requested_as_of_date,
            "current_bullpen_found": current is not None,
            "prior_baseline_found": prior is not None,
            "current_as_of_date": current.as_of_date if current else None,
            "prior_baseline_as_of_date": prior.as_of_date if prior else None,
            "current": current.data if current else {},
            "prior": prior.data if prior else {},
            "source_fingerprints": {
                "current_bullpen_pit": (
                    current.source_fingerprint if current else None
                ),
                "prior_season_bullpen_baseline": (
                    prior.source_fingerprint if prior else None
                ),
            },
            "snapshot_version": self.SNAPSHOT_VERSION,
        }


def aggregate_team_bullpen_metrics(
    team_id: str,
    facts: Iterable[ReliefAppearanceFact],
) -> TeamBullpenMetrics:
    rows = list(facts)
    bf = sum(row.batters_faced for row in rows)
    strikeouts = sum(row.strikeouts for row in rows)
    walks = sum(row.walks for row in rows)
    xwoba_count = sum(row.xwoba_count for row in rows)
    xwoba_numerator = sum(
        float(row.xwoba_against) * row.xwoba_count
        for row in rows
        if row.xwoba_against is not None
    )
    woba_numerator = sum(float(row.woba_numerator or 0.0) for row in rows)
    woba_denominator = sum(float(row.woba_denominator or 0.0) for row in rows)
    contact_count = sum(row.contact_count for row in rows)
    barrel_count = sum(row.barrel_count for row in rows)
    return TeamBullpenMetrics(
        team_id=team_id,
        relief_appearances=len(rows),
        relief_team_games=len({row.game_pk for row in rows}),
        relief_pitch_count=sum(row.pitch_count for row in rows),
        relief_batters_faced=bf,
        strikeouts=strikeouts,
        walks=walks,
        k_pct=_safe_div(strikeouts, bf),
        bb_pct=_safe_div(walks, bf),
        k_minus_bb_pct=(
            round((strikeouts - walks) / bf, 6) if bf > 0 else None
        ),
        xwoba_against=_safe_div(xwoba_numerator, xwoba_count),
        xwoba_count=xwoba_count,
        woba_against=_safe_div(woba_numerator, woba_denominator),
        woba_numerator=(round(woba_numerator, 6) if woba_denominator > 0 else None),
        woba_denominator=(
            round(woba_denominator, 6) if woba_denominator > 0 else None
        ),
        barrel_count=barrel_count,
        contact_count=contact_count,
        barrel_per_contact=_safe_div(barrel_count, contact_count),
        sample_size_status="thin" if bf < SAMPLE_BF_THRESHOLD else "sufficient",
    )


def workload_facts(
    facts: Iterable[ReliefAppearanceFact],
    *,
    cutoff_date: str,
) -> dict[str, Any]:
    rows = list(facts)
    cutoff = date.fromisoformat(cutoff_date)
    output: dict[str, Any] = {}
    for days in (1, 3, 7):
        start = cutoff - timedelta(days=days - 1)
        selected = [
            row for row in rows if start <= date.fromisoformat(row.game_date) <= cutoff
        ]
        output[f"pitches_last_{days}_day" if days == 1 else f"pitches_last_{days}_days"] = sum(
            row.pitch_count for row in selected
        )
        output[
            f"appearances_last_{days}_day"
            if days == 1
            else f"appearances_last_{days}_days"
        ] = len(selected)
        output[
            f"workload_{days}_day_window_start_date"
            if days == 1
            else f"workload_{days}_day_window_start_date"
        ] = start.isoformat()
        output[
            f"workload_{days}_day_window_end_date"
            if days == 1
            else f"workload_{days}_day_window_end_date"
        ] = cutoff.isoformat()

    used_dates = sorted({date.fromisoformat(row.game_date) for row in rows})
    last_used = used_dates[-1] if used_dates else None
    consecutive = 0
    if last_used is not None:
        used_set = set(used_dates)
        cursor = last_used
        while cursor in used_set:
            consecutive += 1
            cursor -= timedelta(days=1)
    output["consecutive_days"] = consecutive
    output["last_used_date"] = last_used.isoformat() if last_used else None
    return output


def _group_facts(
    facts: Iterable[ReliefAppearanceFact],
) -> dict[str, list[ReliefAppearanceFact]]:
    grouped: dict[str, list[ReliefAppearanceFact]] = {}
    for fact in facts:
        grouped.setdefault(fact.team_id, []).append(fact)
    return grouped


def _safe_div(numerator: float | int, denominator: float | int) -> float | None:
    return round(float(numerator) / float(denominator), 6) if denominator else None
