from modules.baseball_module.advanced_pit_enrichment import adapt_unified_pitcher_snapshot
from modules.baseball_module.advanced_pit_enrichment.pitcher_engine_snapshot_adapter import (
    PITCHER_ENGINE_FIELDS,
)
from modules.baseball_module.context_engine.pitcher_engine import PitcherEngine


def _snapshot(**overrides):
    snapshot = {
        "found": True,
        "mlbam_id": 605400,
        "player_name": "Nola, Aaron",
        "siera": 3.21,
        "xfip": 3.44,
        "xera": 3.12,
        "fip": 3.60,
        "k_pct": 0.285,
        "bb_pct": 0.071,
        "ip": 12.2,
        "est_woba": 0.301,
        "brl_percent": 6.5,
        "ev95percent": 34.2,
        "fangraphs_found": True,
        "savant_found": True,
    }
    snapshot.update(overrides)
    return snapshot


def test_adapter_maps_fangraphs_fields_correctly():
    adapted = adapt_unified_pitcher_snapshot(_snapshot())

    assert adapted["name"] == "Nola, Aaron"
    assert adapted["mlbam_id"] == 605400
    assert adapted["siera"] == 3.21
    assert adapted["xfip"] == 3.44
    assert adapted["xera"] == 3.12
    assert adapted["fip"] == 3.60
    assert adapted["k_pct"] == 0.285
    assert adapted["bb_pct"] == 0.071


def test_adapter_maps_savant_fields_correctly():
    adapted = adapt_unified_pitcher_snapshot(_snapshot())

    assert adapted["est_woba"] == 0.301
    assert adapted["brl_percent"] == 6.5
    assert adapted["ev95percent"] == 34.2


def test_adapter_maps_ip_to_innings_pitched():
    adapted = adapt_unified_pitcher_snapshot(_snapshot(ip=14.1, innings_pitched=None))

    assert adapted["ip"] == 14.1
    assert adapted["innings_pitched"] == 14.1


def test_adapter_keeps_existing_innings_pitched_and_backfills_ip():
    adapted = adapt_unified_pitcher_snapshot(_snapshot(ip=None, innings_pitched=15.0))

    assert adapted["innings_pitched"] == 15.0
    assert adapted["ip"] == 15.0


def test_missing_optional_fields_remain_none():
    adapted = adapt_unified_pitcher_snapshot(_snapshot())

    assert adapted["era"] is None
    assert adapted["era_last_5"] is None
    assert adapted["era_trend"] is None
    assert adapted["days_rest"] is None
    assert adapted["last_pitch_count"] is None
    assert adapted["quality_start_pct"] is None
    assert adapted["avg_innings_per_start"] is None
    assert adapted["platoon_splits"] is None
    assert adapted["era_vs_opp"] is None
    assert adapted["ip_vs_opp"] is None
    assert adapted["whip"] is None


def test_adapter_does_not_create_fake_zero_values():
    adapted = adapt_unified_pitcher_snapshot(
        _snapshot(
            siera=None,
            xfip=None,
            fip=None,
            k_pct=None,
            bb_pct=None,
            ip=None,
            innings_pitched=None,
            est_woba=None,
            brl_percent=None,
        )
    )

    assert adapted["siera"] is None
    assert adapted["xfip"] is None
    assert adapted["fip"] is None
    assert adapted["k_pct"] is None
    assert adapted["bb_pct"] is None
    assert adapted["ip"] is None
    assert adapted["innings_pitched"] is None
    assert adapted["est_woba"] is None
    assert adapted["brl_percent"] is None


def test_output_contains_pitcher_engine_contract_keys():
    adapted = adapt_unified_pitcher_snapshot(_snapshot())

    assert set(PITCHER_ENGINE_FIELDS).issubset(adapted.keys())


def test_adapted_output_can_be_passed_to_pitcher_engine_without_keyerror():
    pitcher = adapt_unified_pitcher_snapshot(_snapshot())
    game_data = {
        "pitcher_home": pitcher,
        "pitcher_away": adapt_unified_pitcher_snapshot(_snapshot(mlbam_id=999999, player_name="Other Pitcher")),
        "home_lineup_lhb_pct": 0.45,
        "away_lineup_lhb_pct": 0.45,
    }

    lh, la, metadata = PitcherEngine().adjust_for_pitchers(4.5, 4.2, game_data)

    assert lh > 0
    assert la > 0
    assert "pitcher_home" in metadata
    assert "pitcher_away" in metadata
