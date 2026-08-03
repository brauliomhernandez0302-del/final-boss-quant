"""Leakage-safe relief-appearance facts derived only from local Raw Savant.

Role classification is per team-game. The first distinct pitcher for the
fielding team is the starter/opener and is excluded completely. Every later
distinct pitcher is a relief appearance. Unsafe ordering rejects that
team-game instead of guessing a bullpen role.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .pit_cache import PITCache
from .raw_savant_events_cache import RawSavantBullpenPitch, RawSavantEventsCache


ROLE_RULE_VERSION = "bullpen_first_pitcher_excluded_v1"
FACT_BUILDER_VERSION = "bullpen_relief_appearance_builder_v1"
NON_MLB_TEAM_IDS = frozenset({"AL", "NL"})
STRIKEOUT_EVENTS = frozenset({"strikeout", "strikeout_double_play"})
WALK_EVENTS = frozenset({"walk", "intent_walk"})


class BullpenPITNamespaces:
    RELIEF_APPEARANCE_DAILY = "savant.bullpen.relief_appearance.daily"
    TEAM_BULLPEN_ROLLING = "savant.team_bullpen.rolling"
    TEAM_BULLPEN_PRIOR_BASELINE = "savant.team_bullpen.prior_baseline"


class BullpenPITSources:
    RELIEF_APPEARANCE_DAILY = "baseball_savant_relief_appearance_daily"
    TEAM_BULLPEN_ROLLING = "baseball_savant_team_bullpen_rolling"
    TEAM_BULLPEN_PRIOR_BASELINE = (
        "baseball_savant_team_bullpen_prior_baseline"
    )


@dataclass(frozen=True)
class ReliefAppearanceFact:
    team_id: str
    game_pk: int
    game_date: str
    pitcher_id: int
    starter_opener_id_excluded: int
    pitch_count: int
    batters_faced: int
    strikeouts: int
    walks: int
    xwoba_against: float | None
    xwoba_count: int
    woba_against: float | None
    woba_numerator: float | None
    woba_denominator: float | None
    barrel_count: int
    contact_count: int
    appearance_order: int
    fielding_team_attribution: str
    source_window_start_date: str
    source_window_end_date: str
    source_fingerprint: str
    role_rule_version: str = ROLE_RULE_VERSION
    builder_version: str = FACT_BUILDER_VERSION
    starter_pitch_count_included: int = 0


@dataclass(frozen=True)
class RejectedBullpenTeamGame:
    game_pk: int
    game_date: str
    team_id: str | None
    reason: str
    source_window_start_date: str
    source_window_end_date: str
    source_fingerprint: str
    role_rule_version: str = ROLE_RULE_VERSION


@dataclass(frozen=True)
class ReliefAppearanceBuildResult:
    facts: tuple[ReliefAppearanceFact, ...]
    rejected_team_games: tuple[RejectedBullpenTeamGame, ...]
    accepted_team_games: int
    team_games_seen: int
    rejected_reason_counts: dict[str, int]
    starter_pitches_excluded: int
    starter_pitches_included: int
    duplicate_pitch_contributions_prevented: int
    duplicate_pa_contributions_prevented: int
    excluded_non_mlb_teams: tuple[str, ...]
    rows_processed: int
    source_fingerprint: str


class BullpenReliefAppearanceBuilder:
    """Build and optionally persist daily relief appearance facts."""

    NAMESPACE = BullpenPITNamespaces.RELIEF_APPEARANCE_DAILY
    SOURCE = BullpenPITSources.RELIEF_APPEARANCE_DAILY

    def __init__(
        self,
        *,
        raw_cache_db: Path | str,
        pit_cache_db: Path | str | None = None,
        pit_cache: PITCache | None = None,
        raw_cache: RawSavantEventsCache | None = None,
    ):
        self.raw_cache = raw_cache or RawSavantEventsCache(raw_cache_db)
        self.pit_cache = pit_cache or (
            PITCache(pit_cache_db) if pit_cache_db is not None else None
        )

    def build(self, *, start_date: str, end_date: str) -> ReliefAppearanceBuildResult:
        facts: list[ReliefAppearanceFact] = []
        rejected: list[RejectedBullpenTeamGame] = []
        rejected_counts: Counter[str] = Counter()
        excluded_non_mlb: set[str] = set()
        accepted_team_games = 0
        team_games_seen = 0
        starter_pitches_excluded = 0
        duplicate_pitches = 0
        duplicate_pas = 0
        rows_processed = 0
        input_fingerprints: set[str] = set()

        current_game_pk: int | None = None
        game_rows: list[RawSavantBullpenPitch] = []

        def consume(rows: list[RawSavantBullpenPitch]) -> None:
            nonlocal accepted_team_games, team_games_seen
            nonlocal starter_pitches_excluded, duplicate_pitches, duplicate_pas
            if not rows:
                return
            result = _classify_game(
                rows,
                source_window_start_date=start_date,
                source_window_end_date=end_date,
            )
            facts.extend(result["facts"])
            rejected.extend(result["rejected"])
            rejected_counts.update(item.reason for item in result["rejected"])
            excluded_non_mlb.update(result["excluded_non_mlb_teams"])
            accepted_team_games += result["accepted_team_games"]
            team_games_seen += result["team_games_seen"]
            starter_pitches_excluded += result["starter_pitches_excluded"]
            duplicate_pitches += result["duplicate_pitch_contributions_prevented"]
            duplicate_pas += result["duplicate_pa_contributions_prevented"]

        for pitch in self.raw_cache.iter_bullpen_pitches_by_date_range(
            start_date=start_date,
            end_date=end_date,
        ):
            rows_processed += 1
            input_fingerprints.add(pitch.source_fingerprint)
            if current_game_pk is not None and pitch.game_pk != current_game_pk:
                consume(game_rows)
                game_rows = []
            current_game_pk = pitch.game_pk
            game_rows.append(pitch)
        consume(game_rows)

        source_fingerprint = _fingerprint(
            {
                "source": self.SOURCE,
                "builder_version": FACT_BUILDER_VERSION,
                "role_rule_version": ROLE_RULE_VERSION,
                "start_date": start_date,
                "end_date": end_date,
                "rows_processed": rows_processed,
                "facts": len(facts),
                "accepted_team_games": accepted_team_games,
                "rejected_team_games": len(rejected),
                "input_fingerprints": sorted(input_fingerprints),
            },
            prefix="savant:raw:bullpen:relief-appearance",
        )
        return ReliefAppearanceBuildResult(
            facts=tuple(facts),
            rejected_team_games=tuple(rejected),
            accepted_team_games=accepted_team_games,
            team_games_seen=team_games_seen,
            rejected_reason_counts=dict(sorted(rejected_counts.items())),
            starter_pitches_excluded=starter_pitches_excluded,
            starter_pitches_included=0,
            duplicate_pitch_contributions_prevented=duplicate_pitches,
            duplicate_pa_contributions_prevented=duplicate_pas,
            excluded_non_mlb_teams=tuple(sorted(excluded_non_mlb)),
            rows_processed=rows_processed,
            source_fingerprint=source_fingerprint,
        )

    def persist_daily_facts(
        self,
        *,
        season: int,
        start_date: str,
        end_date: str,
        fetched_at: str | None = None,
    ) -> ReliefAppearanceBuildResult:
        if self.pit_cache is None:
            raise ValueError("pit_cache_db or pit_cache is required for persistence")
        result = self.build(start_date=start_date, end_date=end_date)
        fetched = fetched_at or datetime.now(timezone.utc).isoformat()
        for fact in result.facts:
            payload = asdict(fact)
            payload["record_type"] = "relief_appearance"
            self.pit_cache.save_record(
                namespace=self.NAMESPACE,
                entity_id=f"{fact.team_id}:{fact.game_pk}:{fact.pitcher_id}",
                season=season,
                as_of_date=f"{fact.game_date}T23:59:59Z",
                source=self.SOURCE,
                source_fingerprint=fact.source_fingerprint,
                data=payload,
                fetched_at=fetched,
            )
        for index, item in enumerate(result.rejected_team_games):
            payload = asdict(item)
            payload["record_type"] = "rejected_team_game"
            self.pit_cache.save_record(
                namespace=self.NAMESPACE,
                entity_id=(
                    f"rejected:{item.team_id or 'unknown'}:{item.game_pk}:{index}"
                ),
                season=season,
                as_of_date=f"{item.game_date}T23:59:59Z",
                source=self.SOURCE,
                source_fingerprint=item.source_fingerprint,
                data=payload,
                fetched_at=fetched,
            )
        return result


def fielding_team_for_bullpen_pitch(pitch: RawSavantBullpenPitch) -> str | None:
    """Return the fielding team using inning-half semantics only."""
    home = (pitch.home_team or "").strip()
    away = (pitch.away_team or "").strip()
    half = (pitch.inning_topbot or "").strip().lower()
    if not home or not away or home == away:
        return None
    if half in {"top", "t"}:
        return home
    if half in {"bot", "bottom", "b"}:
        return away
    return None


def _classify_game(
    rows: list[RawSavantBullpenPitch],
    *,
    source_window_start_date: str,
    source_window_end_date: str,
) -> dict[str, Any]:
    first = rows[0]
    game_pk = first.game_pk
    game_date = first.game_date
    contexts = {
        ((row.home_team or "").strip(), (row.away_team or "").strip())
        for row in rows
    }
    known_teams = sorted(
        {
            team
            for context in contexts
            for team in context
            if team and team not in NON_MLB_TEAM_IDS
        }
    )
    if len(contexts) != 1:
        return _reject_game(
            rows,
            teams=known_teams or [None],
            reason="inconsistent_team_context",
            start_date=source_window_start_date,
            end_date=source_window_end_date,
        )
    home, away = next(iter(contexts))
    if not home or not away or home == away:
        return _reject_game(
            rows,
            teams=known_teams or [None],
            reason="invalid_team_context",
            start_date=source_window_start_date,
            end_date=source_window_end_date,
        )
    if home in NON_MLB_TEAM_IDS or away in NON_MLB_TEAM_IDS:
        return {
            "facts": [],
            "rejected": [],
            "accepted_team_games": 0,
            "team_games_seen": 0,
            "starter_pitches_excluded": 0,
            "duplicate_pitch_contributions_prevented": 0,
            "duplicate_pa_contributions_prevented": 0,
            "excluded_non_mlb_teams": {home, away} & NON_MLB_TEAM_IDS,
        }

    grouped: dict[str, list[RawSavantBullpenPitch]] = defaultdict(list)
    for row in rows:
        team = fielding_team_for_bullpen_pitch(row)
        if team is None:
            return _reject_game(
                rows,
                teams=[home, away],
                reason="invalid_inning_half_or_fielding_team",
                start_date=source_window_start_date,
                end_date=source_window_end_date,
            )
        grouped[team].append(row)

    facts: list[ReliefAppearanceFact] = []
    rejected: list[RejectedBullpenTeamGame] = []
    accepted = 0
    starter_excluded = 0
    duplicate_pitches = 0
    duplicate_pas = 0
    for team in (home, away):
        team_rows = grouped.get(team, [])
        if not team_rows:
            continue
        classified = _classify_team_game(
            team_rows,
            team_id=team,
            source_window_start_date=source_window_start_date,
            source_window_end_date=source_window_end_date,
        )
        if classified["rejection"] is not None:
            rejected.append(classified["rejection"])
            continue
        accepted += 1
        facts.extend(classified["facts"])
        starter_excluded += classified["starter_pitches_excluded"]
        duplicate_pitches += classified[
            "duplicate_pitch_contributions_prevented"
        ]
        duplicate_pas += classified["duplicate_pa_contributions_prevented"]

    return {
        "facts": facts,
        "rejected": rejected,
        "accepted_team_games": accepted,
        "team_games_seen": len(grouped),
        "starter_pitches_excluded": starter_excluded,
        "duplicate_pitch_contributions_prevented": duplicate_pitches,
        "duplicate_pa_contributions_prevented": duplicate_pas,
        "excluded_non_mlb_teams": set(),
    }


def _classify_team_game(
    rows: list[RawSavantBullpenPitch],
    *,
    team_id: str,
    source_window_start_date: str,
    source_window_end_date: str,
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda item: (item.at_bat_number, item.pitch_number))
    if any(row.pitcher is None for row in rows):
        return _team_rejection(
            rows,
            team_id,
            "missing_pitcher_id",
            source_window_start_date,
            source_window_end_date,
        )

    unique_rows: list[RawSavantBullpenPitch] = []
    seen_pitch_keys: set[tuple[int, int, int]] = set()
    duplicate_pitches = 0
    for row in rows:
        key = (row.game_pk, row.at_bat_number, row.pitch_number)
        if key in seen_pitch_keys:
            duplicate_pitches += 1
            continue
        seen_pitch_keys.add(key)
        unique_rows.append(row)
    rows = unique_rows

    sequence: list[int] = []
    seen_pitchers: set[int] = set()
    previous: int | None = None
    for row in rows:
        pitcher = int(row.pitcher)  # guarded above
        if pitcher == previous:
            continue
        if pitcher in seen_pitchers:
            return _team_rejection(
                rows,
                team_id,
                "noncontiguous_pitcher_reappearance",
                source_window_start_date,
                source_window_end_date,
            )
        sequence.append(pitcher)
        seen_pitchers.add(pitcher)
        previous = pitcher
    if not sequence:
        return _team_rejection(
            rows,
            team_id,
            "no_confirmed_role_order",
            source_window_start_date,
            source_window_end_date,
        )

    starter_id = sequence[0]
    starter_pitches = sum(1 for row in rows if row.pitcher == starter_id)
    order = {pitcher: index for index, pitcher in enumerate(sequence[1:], start=1)}

    terminal_by_pa: dict[int, RawSavantBullpenPitch] = {}
    terminal_counts: Counter[int] = Counter()
    for row in rows:
        if row.events is not None:
            terminal_counts[row.at_bat_number] += 1
            existing = terminal_by_pa.get(row.at_bat_number)
            if existing is None or row.pitch_number > existing.pitch_number:
                terminal_by_pa[row.at_bat_number] = row
    duplicate_pas = sum(max(count - 1, 0) for count in terminal_counts.values())

    facts: list[ReliefAppearanceFact] = []
    for pitcher_id in sequence[1:]:
        pitcher_rows = [row for row in rows if row.pitcher == pitcher_id]
        terminal_rows = [
            row for row in terminal_by_pa.values() if row.pitcher == pitcher_id
        ]
        xwoba_values = [
            float(row.estimated_woba_using_speedangle)
            for row in terminal_rows
            if row.estimated_woba_using_speedangle is not None
        ]
        woba_numerator = sum(
            float(row.woba_value)
            for row in terminal_rows
            if row.woba_value is not None
        )
        woba_denominator = sum(
            float(row.woba_denom)
            for row in terminal_rows
            if row.woba_denom is not None
        )
        contact_rows = [row for row in terminal_rows if row.launch_speed is not None]
        fact_fingerprint = _fingerprint(
            {
                "source": BullpenPITSources.RELIEF_APPEARANCE_DAILY,
                "role_rule_version": ROLE_RULE_VERSION,
                "team_id": team_id,
                "game_pk": rows[0].game_pk,
                "pitcher_id": pitcher_id,
                "pitch_keys": [
                    [row.at_bat_number, row.pitch_number] for row in pitcher_rows
                ],
                "input_fingerprints": sorted(
                    {row.source_fingerprint for row in pitcher_rows}
                ),
            },
            prefix="savant:raw:bullpen:relief-fact",
        )
        facts.append(
            ReliefAppearanceFact(
                team_id=team_id,
                game_pk=rows[0].game_pk,
                game_date=rows[0].game_date,
                pitcher_id=pitcher_id,
                starter_opener_id_excluded=starter_id,
                pitch_count=len(pitcher_rows),
                batters_faced=len(terminal_rows),
                strikeouts=sum(
                    1 for row in terminal_rows if row.events in STRIKEOUT_EVENTS
                ),
                walks=sum(1 for row in terminal_rows if row.events in WALK_EVENTS),
                xwoba_against=_mean(xwoba_values),
                xwoba_count=len(xwoba_values),
                woba_against=_safe_div(woba_numerator, woba_denominator),
                woba_numerator=(woba_numerator if woba_denominator > 0 else None),
                woba_denominator=(woba_denominator if woba_denominator > 0 else None),
                barrel_count=sum(
                    1 for row in contact_rows if row.launch_speed_angle == 6
                ),
                contact_count=len(contact_rows),
                appearance_order=order[pitcher_id],
                fielding_team_attribution=team_id,
                source_window_start_date=rows[0].game_date,
                source_window_end_date=rows[0].game_date,
                source_fingerprint=fact_fingerprint,
            )
        )

    return {
        "facts": facts,
        "rejection": None,
        "starter_pitches_excluded": starter_pitches,
        "duplicate_pitch_contributions_prevented": duplicate_pitches,
        "duplicate_pa_contributions_prevented": duplicate_pas,
    }


def _team_rejection(
    rows: list[RawSavantBullpenPitch],
    team_id: str,
    reason: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    item = _rejection(rows, team_id, reason, start_date, end_date)
    return {
        "facts": [],
        "rejection": item,
        "starter_pitches_excluded": 0,
        "duplicate_pitch_contributions_prevented": 0,
        "duplicate_pa_contributions_prevented": 0,
    }


def _reject_game(
    rows: list[RawSavantBullpenPitch],
    *,
    teams: Iterable[str | None],
    reason: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    items = [_rejection(rows, team, reason, start_date, end_date) for team in teams]
    return {
        "facts": [],
        "rejected": items,
        "accepted_team_games": 0,
        "team_games_seen": len(items),
        "starter_pitches_excluded": 0,
        "duplicate_pitch_contributions_prevented": 0,
        "duplicate_pa_contributions_prevented": 0,
        "excluded_non_mlb_teams": set(),
    }


def _rejection(
    rows: list[RawSavantBullpenPitch],
    team_id: str | None,
    reason: str,
    start_date: str,
    end_date: str,
) -> RejectedBullpenTeamGame:
    fingerprint = _fingerprint(
        {
            "source": BullpenPITSources.RELIEF_APPEARANCE_DAILY,
            "role_rule_version": ROLE_RULE_VERSION,
            "game_pk": rows[0].game_pk,
            "team_id": team_id,
            "reason": reason,
            "input_fingerprints": sorted({row.source_fingerprint for row in rows}),
        },
        prefix="savant:raw:bullpen:rejected-team-game",
    )
    return RejectedBullpenTeamGame(
        game_pk=rows[0].game_pk,
        game_date=rows[0].game_date,
        team_id=team_id,
        reason=reason,
        source_window_start_date=start_date,
        source_window_end_date=end_date,
        source_fingerprint=fingerprint,
    )


def _fingerprint(payload: dict[str, Any], *, prefix: str) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _safe_div(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator > 0 else None
