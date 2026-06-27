#!/usr/bin/env python
"""Build and validate the isolated contact-adjusted Team Defense PIT layer."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.baseball_module.advanced_pit_enrichment import (  # noqa: E402
    PITCache,
    TeamDefenseDailySnapshotBuilder,
    TeamDefensePITBuilder,
    TeamDefensePriorBaseline,
    adapt_defense_pit_snapshot,
)


DEFAULT_TEAMS = ("ATL", "BOS", "CHC", "HOU", "LAD", "NYY", "SD", "SEA", "TB", "TEX")
DEFAULT_GAME_DATES = ("2024-04-02", "2024-04-05", "2024-04-09")


@dataclass(frozen=True)
class SmokeArgs:
    prior_raw_db: Path
    current_raw_db: Path
    pit_db: Path
    report: Path
    target_season: int = 2024
    prior_season: int = 2023
    prior_start_date: str = "2023-03-30"
    prior_end_date: str = "2023-10-01"
    current_start_date: str = "2024-03-28"
    current_end_date: str = "2024-04-08"
    season_start_date: str = "2024-03-28"
    teams: tuple[str, ...] = DEFAULT_TEAMS
    game_dates: tuple[str, ...] = DEFAULT_GAME_DATES


def run_smoke(args: SmokeArgs) -> dict[str, Any]:
    _validate(args)
    pit_cache = PITCache(args.pit_db)
    prior = TeamDefensePriorBaseline(
        raw_cache_db=args.prior_raw_db,
        pit_cache=pit_cache,
    ).persist_prior_baseline(
        season=args.target_season,
        prior_season=args.prior_season,
        prior_season_start_date=args.prior_start_date,
        prior_season_end_date=args.prior_end_date,
    )

    rolling = TeamDefensePITBuilder(
        raw_cache_db=args.current_raw_db,
        pit_cache=pit_cache,
    )
    current_rows_by_cutoff: dict[str, int] = {}
    attribution_violations: list[dict[str, Any]] = []
    for cutoff_date in _date_range(args.current_start_date, args.current_end_date):
        rows = rolling.persist_cutoff(
            season=args.target_season,
            season_start_date=args.season_start_date,
            as_of_date=cutoff_date,
        )
        current_rows_by_cutoff[cutoff_date] = len(rows)
        invalid = next(iter(rows.values()), {}).get("invalid_fielding_team_rows", 0)
        if invalid:
            attribution_violations.append(
                {"cutoff_date": cutoff_date, "invalid_fielding_team_rows": invalid}
            )

    snapshots = []
    future_violations = []
    same_day_violations = []
    builder = TeamDefenseDailySnapshotBuilder(pit_cache)
    for game_date in args.game_dates:
        for team in args.teams:
            snapshot = builder.build_for_game(
                team_id=team,
                season=args.target_season,
                game_date=game_date,
            )
            adapted = adapt_defense_pit_snapshot(snapshot)
            row = {
                "team_id": team,
                "game_date": game_date,
                "requested_cutoff": snapshot["requested_as_of_date"],
                "current_snapshot_available": snapshot["current_defense_found"],
                "prior_baseline_available": snapshot["prior_baseline_found"],
                "current_as_of_date": snapshot["current_as_of_date"],
                "prior_baseline_as_of_date": snapshot["prior_baseline_as_of_date"],
                "contact_adjusted_defense_proxy": adapted[
                    "contact_adjusted_defense_proxy"
                ],
                "defense_multiplier": adapted["defense_multiplier"],
                "bip_count": adapted["bip_count"],
                "xba_bip_count": adapted["xba_bip_count"],
                "sample_size_status": adapted["sample_size_status"],
                "source_window_start_date": adapted["source_window_start_date"],
                "source_window_end_date": adapted["source_window_end_date"],
                "prior_source_window_start_date": adapted[
                    "prior_source_window_start_date"
                ],
                "prior_source_window_end_date": adapted["prior_source_window_end_date"],
                "provenance_source": adapted["provenance_source"],
                "fallback_used": adapted["fallback_used"],
                "source_fingerprints": adapted["provenance"]["source_fingerprints"],
            }
            snapshots.append(row)
            current_as_of = snapshot.get("current_as_of_date")
            if current_as_of:
                selected = _parse_iso(current_as_of)
                requested = _parse_iso(snapshot["requested_as_of_date"])
                if selected > requested:
                    future_violations.append(row)
                if selected.date() >= date.fromisoformat(game_date):
                    same_day_violations.append(row)

    duplicates = _query_rows(
        args.pit_db,
        """SELECT namespace, entity_id, season, as_of_date, source, COUNT(*) count
           FROM pit_metric_cache GROUP BY 1,2,3,4,5 HAVING COUNT(*) > 1""",
    )
    missing_fingerprints = _query_rows(
        args.pit_db,
        """SELECT namespace, entity_id, season, as_of_date, source
           FROM pit_metric_cache
           WHERE source_fingerprint IS NULL OR trim(source_fingerprint) = ''""",
    )
    pseudo_teams = _query_rows(
        args.pit_db,
        """SELECT namespace, entity_id, season, as_of_date, source
           FROM pit_metric_cache WHERE entity_id IN ('AL', 'NL')""",
    )
    proxies = [float(row["contact_adjusted_defense_proxy"]) for row in snapshots]
    neutral = [
        row for row in snapshots if row["provenance_source"] == "neutral_defense_adjustment"
    ]
    report = {
        "metric_definition": {
            "name": "contact_adjusted_defense_proxy",
            "formula": "mean(actual_out_on_batter - (1 - xBA)) over eligible BIP with xBA",
            "positive_interpretation": "more batter-outs converted than expected from contact quality",
            "raw_fields": {
                "batting_team_attribution": [
                    "batting_team",
                    "bat_team",
                    "batter_team",
                    "team_batting",
                ],
                "fielding_team_attribution": ["home_team", "away_team", "inning_topbot"],
                "batted_ball_outcome": ["events", "bb_type"],
                "estimated_contact_quality": ["estimated_ba_using_speedangle"],
                "actual_result": ["events"],
                "ball_in_play_identification": ["events", "description", "bb_type"],
                "park_game_context": [
                    "game_pk",
                    "game_date",
                    "home_team",
                    "away_team",
                    "hc_x",
                    "hc_y",
                    "hit_location",
                    "if_fielding_alignment",
                    "of_fielding_alignment",
                ],
            },
            "limitations": [
                "not official OAA or DRS",
                "xBA does not encode complete fielder positioning, route, park geometry or opportunity difficulty",
                "binary batter-out treatment does not credit extra outs on every force/double play",
                "BIP without xBA are counted but excluded from the metric numerator and denominator",
            ],
        },
        "build_inputs": {
            "target_season": args.target_season,
            "prior_season": args.prior_season,
            "prior_window": [args.prior_start_date, args.prior_end_date],
            "current_cutoffs": [args.current_start_date, args.current_end_date],
            "teams_sampled": list(args.teams),
            "game_dates": list(args.game_dates),
        },
        "prior_baseline": {
            "rows": len(prior.rows),
            "as_of_date": prior.as_of_date,
            "build_report": prior.build_report,
        },
        "current_rows_by_cutoff": current_rows_by_cutoff,
        "sample_rows": snapshots,
        "neutral_fallbacks": neutral,
        "contact_adjusted_defense_proxy_min_median_max": _min_median_max(proxies),
        "future_violations": future_violations,
        "same_day_violations": same_day_violations,
        "duplicate_pit_keys": duplicates,
        "missing_fingerprints": missing_fingerprints,
        "invalid_fielding_team_attribution": attribution_violations,
        "all_star_pseudo_team_violations": pseudo_teams,
    }
    violation_keys = (
        "future_violations",
        "same_day_violations",
        "duplicate_pit_keys",
        "missing_fingerprints",
        "invalid_fielding_team_attribution",
        "all_star_pseudo_team_violations",
    )
    report["final_classification"] = (
        "FAIL"
        if any(report[key] for key in violation_keys)
        else "PARTIAL"
        if neutral
        else "PASS"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _query_rows(path: Path, sql: str) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql).fetchall()]


def _min_median_max(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": median(values), "max": max(values)}


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _date_range(start: str, end: str) -> list[str]:
    current = date.fromisoformat(start)
    final = date.fromisoformat(end)
    days = []
    while current <= final:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _validate(args: SmokeArgs) -> None:
    for path in (args.prior_raw_db, args.current_raw_db):
        if not path.exists():
            raise FileNotFoundError(path)
    if len(args.teams) < 10:
        raise ValueError("at least 10 teams are required")


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_args(argv: list[str] | None = None) -> SmokeArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-raw-db", required=True, type=Path)
    parser.add_argument("--current-raw-db", required=True, type=Path)
    parser.add_argument("--pit-db", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--teams", default=",".join(DEFAULT_TEAMS))
    parser.add_argument("--game-dates", default=",".join(DEFAULT_GAME_DATES))
    ns = parser.parse_args(argv)
    return SmokeArgs(
        prior_raw_db=ns.prior_raw_db,
        current_raw_db=ns.current_raw_db,
        pit_db=ns.pit_db,
        report=ns.report,
        teams=_csv_tuple(ns.teams),
        game_dates=_csv_tuple(ns.game_dates),
    )


def main(argv: list[str] | None = None) -> int:
    report = run_smoke(_parse_args(argv))
    print(
        "Team Defense PIT smoke "
        f"{report['final_classification']} | "
        f"prior_rows={report['prior_baseline']['rows']} | "
        f"samples={len(report['sample_rows'])} | "
        f"neutral={len(report['neutral_fallbacks'])} | "
        f"proxy={report['contact_adjusted_defense_proxy_min_median_max']}"
    )
    return 0 if report["final_classification"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
