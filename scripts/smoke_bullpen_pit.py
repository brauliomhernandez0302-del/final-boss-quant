#!/usr/bin/env python
"""Build and validate the isolated relief-only Bullpen PIT data layer."""

from __future__ import annotations

import argparse
import json
import resource
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.baseball_module.advanced_pit_enrichment import (  # noqa: E402
    BullpenPriorBaselineBuilder,
    BullpenReliefAppearanceBuilder,
    PITCache,
    TeamBullpenDailySnapshotBuilder,
    TeamBullpenPITBuilder,
    adapt_bullpen_pit_snapshot,
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
    teams: tuple[str, ...] = DEFAULT_TEAMS
    game_dates: tuple[str, ...] = DEFAULT_GAME_DATES


def run_smoke(args: SmokeArgs) -> dict[str, Any]:
    _validate(args)
    pit_cache = PITCache(args.pit_db)

    prior = BullpenPriorBaselineBuilder(
        raw_cache_db=args.prior_raw_db,
        pit_cache=pit_cache,
    ).persist_prior_baseline(
        season=args.target_season,
        prior_season=args.prior_season,
        prior_season_start_date=args.prior_start_date,
        prior_season_end_date=args.prior_end_date,
    )

    current_fact_started = time.perf_counter()
    current_facts = BullpenReliefAppearanceBuilder(
        raw_cache_db=args.current_raw_db,
        pit_cache=pit_cache,
    ).persist_daily_facts(
        season=args.target_season,
        start_date=args.current_start_date,
        end_date=args.current_end_date,
    )
    current_fact_elapsed = time.perf_counter() - current_fact_started

    rolling = TeamBullpenPITBuilder(
        raw_cache_db=args.current_raw_db,
        pit_cache=pit_cache,
    )
    current_rows_by_cutoff: dict[str, int] = {}
    for cutoff_date in _date_range(args.current_start_date, args.current_end_date):
        rows = rolling.persist_cutoff(
            season=args.target_season,
            season_start_date=args.current_start_date,
            as_of_date=cutoff_date,
        )
        current_rows_by_cutoff[cutoff_date] = len(rows)

    snapshot_builder = TeamBullpenDailySnapshotBuilder(pit_cache)
    samples: list[dict[str, Any]] = []
    future_violations: list[dict[str, Any]] = []
    same_day_violations: list[dict[str, Any]] = []
    source_window_violations: list[dict[str, Any]] = []
    missing_fingerprint_violations: list[dict[str, Any]] = []
    hierarchy_coverage: Counter[str] = Counter()
    for game_date in args.game_dates:
        game_day = date.fromisoformat(game_date)
        for team_id in args.teams:
            snapshot = snapshot_builder.build_for_game(
                team_id=team_id,
                season=args.target_season,
                game_date=game_date,
            )
            adapted = adapt_bullpen_pit_snapshot(snapshot)
            hierarchy_coverage[adapted["provenance_source"]] += 1
            row = {
                "team_id": team_id,
                "game_date": game_date,
                "requested_as_of_date": snapshot["requested_as_of_date"],
                "current_snapshot_available": snapshot["current_bullpen_found"],
                "prior_baseline_available": adapted["prior_baseline_available"],
                "current_as_of_date": snapshot["current_as_of_date"],
                "prior_baseline_as_of_date": snapshot["prior_baseline_as_of_date"],
                "provenance_source": adapted["provenance_source"],
                "sample_size_status": adapted["sample_size_status"],
                "quality_metrics": adapted["quality_metrics"],
                "workload_facts": adapted["workload_facts"],
                "source_window_start_date": adapted["source_window_start_date"],
                "source_window_end_date": adapted["source_window_end_date"],
                "source_fingerprints": adapted["source_fingerprints"],
                "neutral_fallback": adapted["neutral_fallback"],
                "applied_multiplier": adapted["applied_multiplier"],
            }
            samples.append(row)

            selected_as_of = adapted.get("selected_as_of_date")
            if selected_as_of:
                selected = _parse_iso(selected_as_of)
                requested = _parse_iso(snapshot["requested_as_of_date"])
                if selected > requested:
                    future_violations.append(row)
                if selected.date() >= game_day:
                    same_day_violations.append(row)
            source_end = adapted.get("source_window_end_date")
            if source_end and date.fromisoformat(source_end) >= game_day:
                source_window_violations.append(row)
            selected_fingerprint_key = {
                "current_bullpen_pit": "current_bullpen_pit",
                "prior_season_bullpen_baseline": (
                    "prior_season_bullpen_baseline"
                ),
            }.get(adapted["provenance_source"])
            if selected_fingerprint_key and not adapted["source_fingerprints"].get(
                selected_fingerprint_key
            ):
                missing_fingerprint_violations.append(row)

    duplicates = _query_rows(
        args.pit_db,
        """SELECT namespace, entity_id, season, as_of_date, source, COUNT(*) count
           FROM pit_metric_cache GROUP BY 1,2,3,4,5 HAVING COUNT(*) > 1""",
    )
    pseudo_teams = _query_rows(
        args.pit_db,
        """SELECT namespace, entity_id, season, as_of_date, source
           FROM pit_metric_cache WHERE entity_id IN ('AL', 'NL')""",
    )
    missing_cache_fingerprints = _query_rows(
        args.pit_db,
        """SELECT namespace, entity_id, season, as_of_date, source
           FROM pit_metric_cache
           WHERE source_fingerprint IS NULL OR trim(source_fingerprint) = ''""",
    )

    quality_distributions = {
        metric: _distribution(
            [
                row["quality_metrics"].get(metric)
                for row in samples
                if row["quality_metrics"].get(metric) is not None
            ]
        )
        for metric in (
            "xwoba_against",
            "woba_against",
            "k_minus_bb_pct",
            "barrel_per_contact",
        )
    }
    workload_distributions = {
        metric: _distribution(
            [
                row["workload_facts"].get(metric)
                for row in samples
                if row["workload_facts"].get(metric) is not None
            ]
        )
        for metric in (
            "pitches_last_1_day",
            "appearances_last_1_day",
            "pitches_last_3_days",
            "appearances_last_3_days",
            "pitches_last_7_days",
            "appearances_last_7_days",
            "consecutive_days",
        )
    }
    bf_distribution = _distribution(
        [
            row["quality_metrics"]["relief_batters_faced"]
            for row in samples
            if row["quality_metrics"].get("relief_batters_faced") is not None
        ]
    )
    neutral = [row for row in samples if row["neutral_fallback"]]
    report = {
        "build_inputs": {
            "target_season": args.target_season,
            "prior_season": args.prior_season,
            "prior_window": [args.prior_start_date, args.prior_end_date],
            "current_window": [args.current_start_date, args.current_end_date],
            "teams_sampled": list(args.teams),
            "game_dates": list(args.game_dates),
            "raw_sources_only": True,
        },
        "prior_baseline": {
            "rows": len(prior.rows),
            "available_rows": sum(
                bool(row["baseline_available"]) for row in prior.rows.values()
            ),
            "as_of_date": prior.as_of_date,
            "build_report": prior.build_report,
        },
        "current_relief_facts": {
            "facts": len(current_facts.facts),
            "accepted_relief_team_games": current_facts.accepted_team_games,
            "rejected_relief_team_games": len(current_facts.rejected_team_games),
            "rejected_reason_counts": current_facts.rejected_reason_counts,
            "starter_pitches_excluded": current_facts.starter_pitches_excluded,
            "starter_pitches_incorrectly_included": (
                current_facts.starter_pitches_included
            ),
            "duplicate_pitch_contributions": (
                current_facts.duplicate_pitch_contributions_prevented
            ),
            "duplicate_pa_contributions": (
                current_facts.duplicate_pa_contributions_prevented
            ),
            "elapsed_sec": round(current_fact_elapsed, 3),
            "max_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                3,
            ),
        },
        "current_rows_by_cutoff": current_rows_by_cutoff,
        "sample_rows": samples,
        "hierarchy_coverage": dict(sorted(hierarchy_coverage.items())),
        "current_snapshot_availability": sum(
            row["current_snapshot_available"] for row in samples
        ),
        "prior_baseline_availability": sum(
            row["prior_baseline_available"] for row in samples
        ),
        "neutral_fallbacks": len(neutral),
        "quality_metric_min_median_max": quality_distributions,
        "bf_min_median_max": bf_distribution,
        "workload_window_min_median_max": workload_distributions,
        "future_violations": future_violations,
        "same_day_violations": same_day_violations,
        "source_window_violations": source_window_violations,
        "missing_fingerprint_violations": (
            missing_fingerprint_violations + missing_cache_fingerprints
        ),
        "duplicate_pit_keys": duplicates,
        "all_star_pseudo_team_violations": pseudo_teams,
        "lambda_integration_enabled": False,
    }
    violation_keys = (
        "future_violations",
        "same_day_violations",
        "source_window_violations",
        "missing_fingerprint_violations",
        "duplicate_pit_keys",
        "all_star_pseudo_team_violations",
    )
    factual_violations = (
        current_facts.starter_pitches_included
        + current_facts.duplicate_pitch_contributions_prevented
        + current_facts.duplicate_pa_contributions_prevented
    )
    report["final_classification"] = (
        "FAIL"
        if factual_violations or any(report[key] for key in violation_keys)
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


def _distribution(values: list[float | int]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    numeric = [float(value) for value in values]
    return {"min": min(numeric), "median": median(numeric), "max": max(numeric)}


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _date_range(start: str, end: str) -> list[str]:
    current = date.fromisoformat(start)
    final = date.fromisoformat(end)
    output = []
    while current <= final:
        output.append(current.isoformat())
        current += timedelta(days=1)
    return output


def _validate(args: SmokeArgs) -> None:
    for path in (args.prior_raw_db, args.current_raw_db):
        if not path.exists():
            raise FileNotFoundError(path)
    if len(args.teams) < 10:
        raise ValueError("at least 10 MLB teams are required")


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
    args = _parse_args(argv)
    report = run_smoke(args)
    current = report["current_relief_facts"]
    print(
        "Bullpen PIT smoke "
        f"{report['final_classification']} | "
        f"accepted={current['accepted_relief_team_games']} | "
        f"rejected={current['rejected_relief_team_games']} | "
        f"coverage={report['hierarchy_coverage']} | "
        f"neutral={report['neutral_fallbacks']} | "
        f"report={args.report}"
    )
    return 0 if report["final_classification"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
