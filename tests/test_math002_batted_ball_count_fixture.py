"""MATH-002 regression fixture — permanent guard against the foul-inflation bug.

Ground truth for these 16 batter-seasons was derived offline in
audit_20260714/math002_diagnostico/ (reporte.md, tabla_comparacion.csv) by
re-counting raw Savant events directly: `bbc_correct` is the intersection of
PA-terminal events (`_is_plate_appearance_event()`) with `launch_speed is not
None`, confirmed to match the live engine's operative definition of a
batted-ball "attempt". Before the 5b fix, the cached `batted_ball_count`
counted any pitch with tracked launch_speed, including non-terminal fouls —
inflated 1.70x-2.19x (mean 1.905x) versus this ground truth.

This test reads the real production cache (data/pit_cache_merged.db), not a
synthetic fixture, because the bug was in the aggregation logic, not in any
one input row — a synthetic fixture could pass while the real rebuild still
had stale/mis-scoped data. It skips (does not fail) when the local data files
are absent, since data/pit_raw/ and *.db caches are gitignored and only exist
in a dev environment with the raw Statcast archives.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modules.baseball_module.advanced_pit_enrichment import PITCache
from modules.baseball_module.advanced_pit_enrichment.tte_daily_snapshot_builder import (
    TTEPITNamespaces,
    TTEPITSources,
)

PIT_CACHE_DB = Path("data/pit_cache_merged.db")

# (season, batter, as_of_date, pa_cached, bbc_correct) — from
# audit_20260714/math002_diagnostico/tabla_comparacion.csv
CASES = [
    (2024, 680776, "2024-09-15", 685, 481),
    (2024, 683002, "2024-09-15", 672, 445),
    (2024, 596019, "2024-09-15", 671, 479),
    (2024, 665742, "2024-09-15", 670, 435),
    (2025, 656941, "2025-09-15", 681, 387),
    (2025, 596019, "2025-09-15", 678, 477),
    (2025, 646240, "2025-09-15", 678, 388),
    (2025, 672695, "2025-09-15", 671, 492),
    (2025, 677587, "2025-09-15", 338, 241),
    (2025, 671289, "2025-09-15", 405, 312),
    (2025, 663886, "2025-09-15", 316, 172),
    (2024, 671056, "2024-09-15", 233, 160),
    (2025, 669257, "2025-09-15", 431, 275),
    (2024, 670032, "2024-09-15", 423, 320),
    (2025, 690987, "2025-09-15", 180, 119),
    (2025, 660162, "2025-09-15", 271, 165),
]

pytestmark = pytest.mark.skipif(
    not PIT_CACHE_DB.exists(),
    reason=f"{PIT_CACHE_DB} not present (gitignored local PIT cache) — MATH-002 fixture requires a dev environment with the real cache built",
)


@pytest.mark.parametrize("season,batter,as_of_date,pa_cached,bbc_correct", CASES)
def test_batted_ball_count_matches_ground_truth(season, batter, as_of_date, pa_cached, bbc_correct):
    cache = PITCache(PIT_CACHE_DB)
    record = cache.get_latest(
        namespace=TTEPITNamespaces.BATTER_ROLLING,
        entity_id=batter,
        season=season,
        as_of_date=as_of_date,
        source=TTEPITSources.BATTER_ROLLING,
    )
    assert record is not None, f"no cached snapshot for batter={batter} season={season} as_of={as_of_date}"
    assert record.data["plate_appearances"] == pa_cached, (
        "plate_appearances drifted from the diagnostic sample — the rebuild "
        "picked up a different snapshot than the one ground-truthed offline"
    )
    assert record.data["batted_ball_count"] == bbc_correct, (
        f"batted_ball_count={record.data['batted_ball_count']} != bbc_correct={bbc_correct} "
        f"for batter={batter} season={season} as_of={as_of_date} — foul-inflation bug "
        "(MATH-002) has regressed, see audit_20260714/math002_diagnostico/reporte.md"
    )
    assert record.data["batted_ball_count"] <= record.data["plate_appearances"], (
        "batted_ball_count must never exceed plate_appearances by definition"
    )
