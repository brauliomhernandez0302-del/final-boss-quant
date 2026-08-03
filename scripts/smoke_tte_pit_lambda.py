#!/usr/bin/env python
"""Reproducible Team/TTE PIT lambda smoke harness.

This script is intentionally isolated from live and backtest paths. It builds
or reuses experimental PIT cache rows, samples Team/TTE snapshots with a D-1
cutoff, adapts them to lambda, and writes a validation report.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
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
    PITCache,
    RawSavantEventsCache,
    SavantOffenseRollingBuilder,
    TTEDailySnapshotBuilder,
    TTEPITNamespaces,
    TTEPITSources,
    TTEPriorBaselineBuilder,
    adapt_tte_pit_snapshot_to_lambda,
    previous_day_cutoff_for_game_date,
)
from modules.baseball_module.advanced_pit_enrichment.savant_offense_daily_aggregator import (  # noqa: E402
    METRIC_VERSION,
    _inputs_fingerprint,
    _rolling_payload,
)
from modules.baseball_module.advanced_pit_enrichment.tte_pit_adapter import (  # noqa: E402
    LAMBDA_MAX,
    LAMBDA_MIN,
)


NON_MLB_TEAM_IDS = frozenset({"AL", "NL"})
WEIGHT_SUM_TOLERANCE = 0.01


@dataclass(frozen=True)
class SmokeArgs:
    prior_raw_db: Path
    current_raw_db: Path
    pit_db: Path
    report: Path
    target_season: int
    prior_season: int
    current_start_date: str
    current_end_date: str
    sample_game_dates: list[str]
    sample_team_ids: list[str]
    season_start_date: str


def run_smoke(args: SmokeArgs) -> dict[str, Any]:
    _validate_inputs(args)
    pit_cache = PITCache(args.pit_db)
    prior_raw = RawSavantEventsCache(args.prior_raw_db)
    current_raw = RawSavantEventsCache(args.current_raw_db)

    prior_window = _season_window(prior_raw, args.prior_season)
    prior_result = TTEPriorBaselineBuilder(
        raw_cache_db=args.prior_raw_db,
        pit_cache=pit_cache,
    ).persist_prior_baseline(
        season=args.target_season,
        prior_season=args.prior_season,
        prior_season_start_date=prior_window["start_date"],
        prior_season_end_date=prior_window["end_date"],
    )

    current_rows_by_cutoff: dict[str, int] = {}
    for cutoff_date in _date_range(args.current_start_date, args.current_end_date):
        rows = _persist_current_team_offense_cutoff(
            raw_cache=current_raw,
            pit_cache=pit_cache,
            season=args.target_season,
            season_start_date=args.season_start_date,
            as_of_date=cutoff_date,
        )
        current_rows_by_cutoff[f"{cutoff_date}T00:00:00+00:00"] = len(rows)

    sample_rows = _sample_lambda_rows(
        pit_cache=pit_cache,
        target_season=args.target_season,
        game_dates=args.sample_game_dates,
        team_ids=args.sample_team_ids,
    )
    violations = _collect_violations(
        pit_db=args.pit_db,
        target_season=args.target_season,
        prior_season=args.prior_season,
        sample_rows=sample_rows,
    )
    fallback_reasons = Counter(
        str(row["fallback_used"]) for row in sample_rows if row.get("fallback_used")
    )
    lambda_values = [row["lambda_offense"] for row in sample_rows if row["lambda_offense"] is not None]
    current_weights = [
        row["blend_current_weight"]
        for row in sample_rows
        if row.get("blend_current_weight") is not None
    ]
    prior_weights = [
        row["blend_prior_weight"]
        for row in sample_rows
        if row.get("blend_prior_weight") is not None
    ]

    report = {
        "build_inputs": {
            "target_season": args.target_season,
            "prior_season": args.prior_season,
            "prior_window": prior_window,
            "current_start_date": args.current_start_date,
            "current_end_date": args.current_end_date,
            "season_start_date": args.season_start_date,
            "sample_game_dates": args.sample_game_dates,
            "sample_team_ids": args.sample_team_ids,
        },
        "db_paths": {
            "prior_raw_db": str(args.prior_raw_db),
            "current_raw_db": str(args.current_raw_db),
            "pit_db": str(args.pit_db),
            "report": str(args.report),
        },
        "raw_event_counts": {
            "prior_window": prior_raw.count_events_by_date_range(
                start_date=prior_window["start_date"],
                end_date=prior_window["end_date"],
            ),
            "current_window": current_raw.count_events_by_date_range(
                start_date=args.current_start_date,
                end_date=args.current_end_date,
            ),
        },
        "prior_baseline_rows": {
            "count": len(prior_result.rows),
            "as_of_date": prior_result.as_of_date,
            "build_report": prior_result.build_report,
        },
        "current_pit_rows_by_cutoff": current_rows_by_cutoff,
        "sample_rows": sample_rows,
        "lambda_min_median_max": _min_median_max(lambda_values),
        "current_weight_min_median_max": _min_median_max(current_weights),
        "prior_weight_min_median_max": _min_median_max(prior_weights),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        **violations,
    }
    report["final_classification"] = _classify(report)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _persist_current_team_offense_cutoff(
    *,
    raw_cache: RawSavantEventsCache,
    pit_cache: PITCache,
    season: int,
    season_start_date: str,
    as_of_date: str,
) -> dict[str, dict[str, Any]]:
    builder = SavantOffenseRollingBuilder(cache=raw_cache)
    result = builder.build_teams_for_as_of_date(
        season_start_date=season_start_date,
        as_of_date=as_of_date,
    )
    fingerprint = _inputs_fingerprint(
        raw_cache=raw_cache,
        season=season,
        season_start_date=season_start_date,
        as_of_date=as_of_date,
        metric_version=METRIC_VERSION,
        entity="team",
    )
    fetched_at = datetime.now(timezone.utc).isoformat()
    persisted: dict[str, dict[str, Any]] = {}
    for team_id, metrics in result.rows.items():
        if str(team_id) in NON_MLB_TEAM_IDS:
            continue
        data = _rolling_payload(
            metrics,
            source_window_start_date=season_start_date,
            source_window_end_date=as_of_date,
            metric_version=METRIC_VERSION,
        )
        data["missing_batting_team_rows"] = result.missing_team_rows
        pit_cache.save_record(
            namespace=TTEPITNamespaces.TEAM_OFFENSE_ROLLING,
            entity_id=team_id,
            season=season,
            as_of_date=f"{as_of_date}T00:00:00Z",
            source=TTEPITSources.TEAM_OFFENSE_ROLLING,
            source_fingerprint=fingerprint,
            data=data,
            fetched_at=fetched_at,
        )
        persisted[str(team_id)] = data
    return persisted


def _sample_lambda_rows(
    *,
    pit_cache: PITCache,
    target_season: int,
    game_dates: list[str],
    team_ids: list[str],
) -> list[dict[str, Any]]:
    builder = TTEDailySnapshotBuilder(pit_cache)
    rows: list[dict[str, Any]] = []
    for game_date in game_dates:
        expected_cutoff = previous_day_cutoff_for_game_date(game_date)
        for team_id in team_ids:
            snapshot = builder.build_for_game(
                team_id=team_id,
                season=target_season,
                game_date=game_date,
            )
            adapted = adapt_tte_pit_snapshot_to_lambda(snapshot)
            rows.append(
                {
                    "game_date": game_date,
                    "team_id": str(team_id),
                    "expected_requested_as_of_date": expected_cutoff,
                    "requested_as_of_date": snapshot.get("requested_as_of_date"),
                    "team_offense_as_of_date": snapshot.get("team_offense_as_of_date"),
                    "prior_baseline_as_of_date": snapshot.get("prior_baseline_as_of_date"),
                    "team_rolling_found": snapshot.get("team_rolling_found"),
                    "prior_baseline_found": snapshot.get("prior_baseline_found"),
                    "lambda_offense": adapted.get("lambda_offense"),
                    "blend_current_weight": adapted.get("blend_current_weight"),
                    "blend_prior_weight": adapted.get("blend_prior_weight"),
                    "sample_size_status": adapted.get("sample_size_status"),
                    "fallback_used": adapted.get("fallback_used"),
                    "missing_inputs": adapted.get("provenance", {}).get("missing_inputs", []),
                    "source_fingerprints": snapshot.get("source_fingerprints", {}),
                }
            )
    return rows


def _collect_violations(
    *,
    pit_db: Path,
    target_season: int,
    prior_season: int,
    sample_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    pit_rows = _load_pit_rows(pit_db)
    current_rows = [
        row
        for row in pit_rows
        if row["namespace"] == TTEPITNamespaces.TEAM_OFFENSE_ROLLING
        and row["source"] == TTEPITSources.TEAM_OFFENSE_ROLLING
        and row["season"] == target_season
    ]
    prior_rows = [
        row
        for row in pit_rows
        if row["namespace"] == TTEPITNamespaces.TEAM_OFFENSE_PRIOR_BASELINE
        and row["source"] == TTEPITSources.TEAM_OFFENSE_PRIOR_BASELINE
        and row["season"] == target_season
    ]

    violations = {
        "future_violations": [],
        "same_day_or_later_violations": [],
        "source_window_violations": [],
        "duplicate_key_violations": _duplicate_key_violations(pit_db),
        "missing_fingerprint_violations": [],
        "fake_zero_violations": [],
        "lambda_clamp_violations": [],
        "weight_sum_violations": [],
        "prior_season_violations": [],
        "all_star_pseudo_team_violations": [],
        "cutoff_policy_violations": [],
        "missing_input_contract_violations": [],
    }

    for row in current_rows + prior_rows:
        if not row["source_fingerprint"]:
            violations["missing_fingerprint_violations"].append(_row_ref(row))
        if str(row["entity_id"]) in NON_MLB_TEAM_IDS:
            violations["all_star_pseudo_team_violations"].append(_row_ref(row))

    for row in current_rows:
        data = row["data"]
        source_end = data.get("source_window_end_date")
        if source_end and _date_from_iso(row["as_of_date"]) and source_end > _date_from_iso(row["as_of_date"]):
            item = _row_ref(row)
            item["source_window_end_date"] = source_end
            violations["source_window_violations"].append(item)
        violations["fake_zero_violations"].extend(_fake_zero_violations(row))

    for row in prior_rows:
        data = row["data"]
        if data.get("prior_season") != prior_season:
            item = _row_ref(row)
            item["prior_season"] = data.get("prior_season")
            violations["prior_season_violations"].append(item)
        start = str(data.get("source_window_start_date", ""))
        end = str(data.get("source_window_end_date", ""))
        if (start and not start.startswith(str(prior_season))) or (
            end and not end.startswith(str(prior_season))
        ):
            item = _row_ref(row)
            item["source_window_start_date"] = start
            item["source_window_end_date"] = end
            violations["prior_season_violations"].append(item)
        violations["fake_zero_violations"].extend(_fake_zero_violations(row))

    current_by_team = {}
    for row in current_rows:
        current_by_team.setdefault(str(row["entity_id"]), []).append(row)

    for sample in sample_rows:
        if sample["requested_as_of_date"] != sample["expected_requested_as_of_date"]:
            violations["cutoff_policy_violations"].append(sample)
        lambda_value = sample.get("lambda_offense")
        if lambda_value is not None and not (LAMBDA_MIN <= float(lambda_value) <= LAMBDA_MAX):
            violations["lambda_clamp_violations"].append(sample)
        cur_w = sample.get("blend_current_weight")
        prior_w = sample.get("blend_prior_weight")
        if cur_w is not None and prior_w is not None:
            weight_sum = float(cur_w) + float(prior_w)
            if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
                item = dict(sample)
                item["weight_sum"] = weight_sum
                violations["weight_sum_violations"].append(item)
        if sample.get("missing_inputs") and sample.get("fallback_used") is None and lambda_value is not None:
            violations["missing_input_contract_violations"].append(sample)

        cutoff = _normalize_iso(sample["requested_as_of_date"])
        game_day = sample["game_date"]
        for row in current_by_team.get(str(sample["team_id"]), []):
            if row["as_of_date"] > cutoff:
                item = _row_ref(row)
                item["requested_as_of_date"] = cutoff
                item["game_date"] = game_day
                violations["future_violations"].append(item)
            if _date_from_iso(row["as_of_date"]) >= game_day:
                item = _row_ref(row)
                item["game_date"] = game_day
                violations["same_day_or_later_violations"].append(item)

    return violations


def _classify(report: dict[str, Any]) -> str:
    violation_keys = [
        "future_violations",
        "same_day_or_later_violations",
        "source_window_violations",
        "duplicate_key_violations",
        "missing_fingerprint_violations",
        "fake_zero_violations",
        "lambda_clamp_violations",
        "weight_sum_violations",
        "prior_season_violations",
        "all_star_pseudo_team_violations",
        "cutoff_policy_violations",
        "missing_input_contract_violations",
    ]
    if any(report[key] for key in violation_keys):
        return "FAIL"
    if report["fallback_reasons"] or any(row["lambda_offense"] is None for row in report["sample_rows"]):
        return "PARTIAL"
    return "PASS"


def _season_window(raw_cache: RawSavantEventsCache, season: int) -> dict[str, str]:
    with sqlite3.connect(raw_cache.db_path) as conn:
        row = conn.execute(
            """
            SELECT MIN(game_date), MAX(game_date)
            FROM raw_savant_events
            WHERE game_date >= ? AND game_date <= ?
            """,
            (f"{season}-01-01", f"{season}-12-31"),
        ).fetchone()
    if row is None or row[0] is None or row[1] is None:
        raise ValueError(f"no raw events found for prior_season={season}")
    return {"start_date": row[0], "end_date": row[1]}


def _load_pit_rows(pit_db: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(pit_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT namespace, entity_id, season, as_of_date, source, source_fingerprint, data_json
            FROM pit_metric_cache
            ORDER BY namespace, entity_id, season, as_of_date, source
            """
        ).fetchall()
    return [
        {
            "namespace": row["namespace"],
            "entity_id": row["entity_id"],
            "season": int(row["season"]),
            "as_of_date": row["as_of_date"],
            "source": row["source"],
            "source_fingerprint": row["source_fingerprint"],
            "data": json.loads(row["data_json"]),
        }
        for row in rows
    ]


def _duplicate_key_violations(pit_db: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(pit_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT namespace, entity_id, season, as_of_date, source, COUNT(*) AS count
            FROM pit_metric_cache
            GROUP BY namespace, entity_id, season, as_of_date, source
            HAVING COUNT(*) > 1
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _fake_zero_violations(row: dict[str, Any]) -> list[dict[str, Any]]:
    data = row["data"]
    checks = [
        ("est_woba", "pa"),
        ("woba", "woba_denominator"),
        ("barrel_pa", "pa"),
        ("bb_pct", "pa"),
        ("k_pct", "pa"),
        ("brl_percent", "batted_ball_count"),
        ("ev95percent", "batted_ball_count"),
        ("team_est_woba_prior", "pa_prior"),
        ("team_woba_prior", "pa_prior"),
        ("barrel_pa_prior", "pa_prior"),
        ("bb_pct_prior", "pa_prior"),
        ("k_pct_prior", "pa_prior"),
        ("brl_percent_prior", "bip_prior"),
        ("ev95percent_prior", "bip_prior"),
    ]
    violations = []
    for metric, denominator in checks:
        if data.get(metric) == 0 and data.get(denominator) in (None, 0):
            item = _row_ref(row)
            item["metric"] = metric
            item["denominator"] = denominator
            violations.append(item)
    return violations


def _row_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "namespace": row["namespace"],
        "entity_id": row["entity_id"],
        "season": row["season"],
        "as_of_date": row["as_of_date"],
        "source": row["source"],
    }


def _min_median_max(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    ordered = sorted(float(value) for value in values)
    return {"min": ordered[0], "median": float(median(ordered)), "max": ordered[-1]}


def _date_range(start_date: str, end_date: str) -> list[str]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise ValueError("current_end_date must be on or after current_start_date")
    days = []
    current = start
    while current <= end:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _normalize_iso(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _date_from_iso(value: str | None) -> str | None:
    if not value:
        return None
    return _normalize_iso(value)[:10]


def _validate_inputs(args: SmokeArgs) -> None:
    for path in (args.prior_raw_db, args.current_raw_db):
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.sample_game_dates:
        raise ValueError("--sample-game-dates is required")
    if not args.sample_team_ids:
        raise ValueError("--sample-team-ids is required")
    for raw_date in [
        args.current_start_date,
        args.current_end_date,
        args.season_start_date,
        *args.sample_game_dates,
    ]:
        date.fromisoformat(raw_date)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_args(argv: list[str] | None = None) -> SmokeArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-raw-db", required=True, type=Path)
    parser.add_argument("--current-raw-db", required=True, type=Path)
    parser.add_argument("--pit-db", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--target-season", required=True, type=int)
    parser.add_argument("--prior-season", required=True, type=int)
    parser.add_argument("--current-start-date", required=True)
    parser.add_argument("--current-end-date", required=True)
    parser.add_argument("--sample-game-dates", required=True)
    parser.add_argument("--sample-team-ids", required=True)
    parser.add_argument("--season-start-date", required=True)
    ns = parser.parse_args(argv)
    return SmokeArgs(
        prior_raw_db=ns.prior_raw_db,
        current_raw_db=ns.current_raw_db,
        pit_db=ns.pit_db,
        report=ns.report,
        target_season=ns.target_season,
        prior_season=ns.prior_season,
        current_start_date=ns.current_start_date,
        current_end_date=ns.current_end_date,
        sample_game_dates=_parse_csv(ns.sample_game_dates),
        sample_team_ids=_parse_csv(ns.sample_team_ids),
        season_start_date=ns.season_start_date,
    )


def main(argv: list[str] | None = None) -> int:
    report = run_smoke(_parse_args(argv))
    print(
        "TTE PIT smoke "
        f"{report['final_classification']} | "
        f"samples={len(report['sample_rows'])} | "
        f"lambda={report['lambda_min_median_max']} | "
        f"fallbacks={report['fallback_reasons']} | "
        f"report={report['db_paths']['report']}"
    )
    return 0 if report["final_classification"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
