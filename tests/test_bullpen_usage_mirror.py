"""`fetch_bullpen_usage` must mirror what the bullpen engine actually did.

The dashboard used to show three numbers that are different by construction and
looked like they measured the same thing (VAL-7.2): a roster list, the engine's
`n_pitchers` (Savant coverage) and its `n_siera_pitchers` (FanGraphs coverage).
The API now rebuilds the engine's contributor set instead — which is only worth
anything if it can't drift from the engine.

So these tests run the REAL aggregators (`_aggregate_team_savant`,
`_aggregate_team_siera`) over the same inputs the API is fed, and assert the
counts agree. If someone changes what the engine counts, this fails.
"""
import pytest

from api import mlb_presentation as mp
from modules.baseball_module.context_engine.bullpen_engine import (
    _aggregate_team_savant,
    _aggregate_team_siera,
)

TEAM_ID = 112
SEASON = 2026

# A roster shaped like the real one: two full-coverage relievers, one Savant-only,
# one FanGraphs-only, a position player with a single mop-up inning, a reliever
# with no data at all, and a starter (not in reliever_ids).
ROSTER = {
    1: "Full Coverage",
    2: "Also Full",
    3: "Savant Only",
    4: "Fangraphs Only",
    5: "Catcher Who Pitched",
    6: "No Data At All",
    7: "Rotation Starter",
}
RELIEVER_IDS = {1, 2, 3, 4, 5, 6}
SAVANT_EXP = {
    1: {"pa": 189.0, "est_woba": 0.300},
    2: {"pa": 149.0, "est_woba": 0.310},
    3: {"pa": 136.0, "est_woba": 0.290},
    7: {"pa": 500.0, "est_woba": 0.320},   # starter — must never be counted
}
FG_PITCHERS = {
    1: {"siera": 3.50, "ip": 44.1},
    2: {"xfip": 3.90, "ip": 35.2},         # xFIP fallback path
    4: {"siera": 4.10, "ip": 16.2},
    5: {"siera": 9.99, "ip": 1.0},         # the catcher's one inning
    7: {"siera": 3.10, "ip": 150.0},
}
POSITIONS = {1: "P", 2: "P", 3: "P", 4: "P", 5: "C", 6: "P", 7: "P"}


@pytest.fixture
def usage(monkeypatch):
    monkeypatch.setattr(mp, "_current_mlb_season", lambda: SEASON)
    monkeypatch.setattr(mp, "_fetch_team_roster", lambda tid, s: ROSTER)
    monkeypatch.setattr(mp, "_fetch_reliever_ids", lambda tid, s, r: RELIEVER_IDS)
    monkeypatch.setattr(mp, "_season_pitcher_sources", lambda s: (SAVANT_EXP, FG_PITCHERS))
    monkeypatch.setattr(mp, "_fetch_roster_positions", lambda tid, s: POSITIONS)
    return mp.fetch_bullpen_usage(TEAM_ID)


def test_counts_equal_the_engines_own(usage):
    """The whole point: n_savant/n_siera reproduce the engine's counts exactly."""
    savant = _aggregate_team_savant(ROSTER, SAVANT_EXP, {}, RELIEVER_IDS)
    siera = _aggregate_team_siera(ROSTER, FG_PITCHERS, RELIEVER_IDS)

    assert usage["n_savant"] == savant["n_pitchers"] == 3
    assert usage["n_siera"] == siera["n_pitchers"] == 4


def test_single_count_is_the_list_length(usage):
    assert usage["n_used"] == len(usage["pitchers"]) == 5
    assert usage["n_classified"] == 6  # the zero-data reliever is classified, not used


def test_zero_weight_reliever_is_not_listed(usage):
    names = [p["name"] for p in usage["pitchers"]]
    assert "No Data At All" not in names


def test_starter_never_appears_even_with_the_biggest_sample(usage):
    """The Savant map has him at 500 PA — reliever_ids is what keeps him out,
    exactly as it does inside the engine's aggregate."""
    assert "Rotation Starter" not in [p["name"] for p in usage["pitchers"]]


def test_position_player_is_listed_with_his_real_weight_not_hidden(usage):
    """He IS in the engine's SIERA aggregate (1.0 IP), so hiding him would make
    the UI disagree with the model. He's labeled and carries his true weight."""
    catcher = next(p for p in usage["pitchers"] if p["name"] == "Catcher Who Pitched")
    assert catcher["position"] == "C"
    assert catcher["siera_ip"] == 1.0
    assert catcher["savant_pa"] is None


def test_sorted_by_weight(usage):
    pas = [p["savant_pa"] or 0.0 for p in usage["pitchers"]]
    assert pas == sorted(pas, reverse=True)


def test_degrades_loudly_when_role_classification_fails(monkeypatch):
    monkeypatch.setattr(mp, "_current_mlb_season", lambda: SEASON)
    monkeypatch.setattr(mp, "_fetch_team_roster", lambda tid, s: ROSTER)
    monkeypatch.setattr(mp, "_fetch_reliever_ids", lambda tid, s, r: None)

    out = mp.fetch_bullpen_usage(TEAM_ID)
    assert out["degraded"] is True
    assert out["pitchers"] == []
    assert out["n_used"] == 0
