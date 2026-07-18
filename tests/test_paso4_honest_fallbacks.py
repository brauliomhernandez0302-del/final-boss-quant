"""
Roadmap Step 4 regression tests (audit_20260714/14_remediation_roadmap.md) —
FALL-001 + FALL-002 + the stadium-dictionary consistency test the audit's
own findings.csv flagged as missing under REG-015.

Principle (see CONTRACTS.md): a fallback never produces a value
indistinguishable from a real measurement. Missing is signaled with
explicit provenance; it is never replaced by a fabricated plausible value.

FALL-001: park_weather_engine.py's weather multiplier stays numerically
identical when weather data is absent — only a new `weather_source`
("live"|"missing") metadata field is added.

FALL-002: data_fetchers.py::get_travel_fatigue() no longer fabricates
1000mi/1tz for an unmapped venue; it reports `travel_source="missing"`
with honest 0 values, and hfa_engine.py applies a neutral (no) travel
penalty when it sees that flag instead of guessing.

Stadium consistency: every venue name in park_weather_engine.STADIUM_DATABASE
must resolve in BOTH data_fetchers.WeatherAPI.STADIUM_COORDS and
historical_weather._STADIUM_COORDS — the exact invariant REG-015 violated
for 4 renamed stadiums, undetected for weeks. Had this test existed then,
it would have caught it in CI before any production data was silently
degraded.
"""
import importlib.util
from pathlib import Path

import pytest

from data_fetchers import WeatherAPI
from modules.baseball_module.hfa.park_weather_engine import (
    STADIUM_DATABASE,
    ParkWeatherEngine,
)
from modules.baseball_module.hfa.hfa_engine import HFAEngine

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_historical_weather_coords():
    spec = importlib.util.spec_from_file_location(
        "historical_weather_test_import",
        REPO_ROOT / "modules" / "baseball_module" / "hfa" / "historical_weather.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._STADIUM_COORDS


# ── PARTE 3: consistencia de los 3 diccionarios de estadios ────────────────


def test_every_park_database_venue_resolves_in_weather_and_historical_coords():
    """The exact invariant REG-015 broke: STADIUM_DATABASE (park factors)
    gained 4 renamed-venue keys (Rate Field, Oriole Park at Camden Yards,
    Daikin Park, UNIQLO Field at Dodger Stadium) that WeatherAPI.
    STADIUM_COORDS / historical_weather._STADIUM_COORDS didn't have —
    silently dropping live weather + corrupting travel-distance for those
    teams' home games for weeks, undetected. Asserts the subset invariant
    directly rather than re-testing individual venue names, so the NEXT
    rename trips this test the same way.
    """
    historical_coords = _load_historical_weather_coords()
    park_venues = set(STADIUM_DATABASE.keys())
    weather_venues = set(WeatherAPI.STADIUM_COORDS.keys())
    historical_venues = set(historical_coords.keys())

    missing_from_weather = park_venues - weather_venues
    missing_from_historical = park_venues - historical_venues

    assert not missing_from_weather, (
        f"venues in STADIUM_DATABASE but missing from WeatherAPI.STADIUM_COORDS "
        f"(live weather/travel would silently go neutral for these): {sorted(missing_from_weather)}"
    )
    assert not missing_from_historical, (
        f"venues in STADIUM_DATABASE but missing from historical_weather._STADIUM_COORDS: "
        f"{sorted(missing_from_historical)}"
    )


# Known, legitimate extra aliases that exist in one coordinate dict but are
# not (and don't need to be) canonical STADIUM_DATABASE keys — documented
# here, not silenced, per the task's own instruction not to skip silently.
#   "Rogers Center" (WeatherAPI.STADIUM_COORDS only): a pre-existing
#   American-spelling duplicate of the canonical "Rogers Centre" — extra,
#   harmless, does not violate the subset invariant above (it's not a gap,
#   it's a superfluous synonym). Not removed here — out of this step's
#   scope (not one of FALL-001/FALL-002/the missing-test gap).
_KNOWN_EXTRA_ALIASES = {"WeatherAPI.STADIUM_COORDS": {"Rogers Center"}}


def test_known_extra_aliases_are_still_just_that_and_nothing_more():
    """Guards the documented exception list itself: if 'Rogers Center'
    either disappears (fine) or a NEW undocumented extra key shows up
    (worth a human look, not necessarily a bug), this test tells you which."""
    weather_venues = set(WeatherAPI.STADIUM_COORDS.keys())
    park_venues = set(STADIUM_DATABASE.keys())
    extras = weather_venues - park_venues
    assert extras == _KNOWN_EXTRA_ALIASES["WeatherAPI.STADIUM_COORDS"], (
        f"WeatherAPI.STADIUM_COORDS has undocumented extra venue(s) not in "
        f"STADIUM_DATABASE: {sorted(extras - _KNOWN_EXTRA_ALIASES['WeatherAPI.STADIUM_COORDS'])}. "
        f"Update _KNOWN_EXTRA_ALIASES above if this is legitimate."
    )


# ── FALL-001: weather provenance ────────────────────────────────────────


def test_unmapped_venue_weather_source_is_missing_and_multiplier_unchanged():
    """An unmapped venue name -> weather_source='missing', and the
    multiplier is EXACTLY the pre-fix neutral value (1.0 total_mult *
    whatever park_mult defaults to for an unknown park — 1.00) — proof
    this fix is purely additive metadata, zero numeric change."""
    engine = ParkWeatherEngine()
    game_data = {
        "park": {"name": "Totally Fictional Stadium"},
        "weather": {},  # no weather data
    }
    lh, la, meta = engine.adjust_for_park_and_weather(4.5, 4.0, game_data)

    assert meta["weather_source"] == "missing"
    assert meta["weather_mult"] == 1.0
    assert meta["park_factor"] == 1.00
    assert lh == 4.5  # total_mult=1.0 -> unchanged
    assert la == 4.0


def test_mapped_venue_with_real_weather_source_is_live():
    """A known venue with real weather data present -> weather_source='live'."""
    engine = ParkWeatherEngine()
    game_data = {
        "park": {"name": "Fenway Park"},
        "weather": {
            "temp_f": 75.0, "wind_speed_mph": 5.0, "wind_direction": 90.0,
            "rain_mm": 0.0, "conditions": "Clear",
        },
    }
    lh, la, meta = engine.adjust_for_park_and_weather(4.5, 4.0, game_data)

    assert meta["weather_source"] == "live"
    # Real park factor for Fenway (1.04) must still apply — untouched by this fix.
    assert meta["park_factor"] == pytest.approx(1.04)


def test_backtest_style_no_weather_key_at_all_is_also_missing():
    """The backtest never populates game_data['weather'] at all (weather-
    blind by design) — confirms it will report 'missing' on every game,
    exactly as the task describes as the correct, honest, expected
    collateral effect (not a bug to fix)."""
    engine = ParkWeatherEngine()
    game_data = {"park": {"name": "Wrigley Field"}}  # no "weather" key
    _, _, meta = engine.adjust_for_park_and_weather(4.5, 4.0, game_data)
    assert meta["weather_source"] == "missing"


# ── FALL-002: travel provenance ─────────────────────────────────────────


def test_hfa_engine_applies_neutral_when_travel_source_missing():
    """travel_source_away='missing' -> zero travel penalty (neutral),
    regardless of whatever miles/time_zones values happen to be present —
    the whole point is not trusting fabricated numbers even if they leak
    through."""
    engine = HFAEngine()
    game_data = {
        "park": {"name": "Yankee Stadium"},
        "travel_source_away": "missing",
        # Deliberately populated with what the OLD fabricated fallback used
        # to produce (1000mi/1tz) to prove the new code ignores it.
        "miles_traveled_away": 1000,
        "time_zones_crossed_away": 1,
    }
    lh, la, meta = engine.get_adjusted_lambdas(4.5, 4.0, game_data)

    assert meta["travel_source"] == "missing"
    assert meta["travel_penalty"] == 0.0
    assert la == 4.0  # unchanged — no penalty applied


def test_hfa_engine_applies_real_penalty_when_travel_source_live():
    """travel_source_away='live' (or absent, the default) with real
    cross-country travel data still applies the normal, unmodified
    penalty formula — proof FALL-002 didn't neuter real travel fatigue,
    only the fabricated-fallback case."""
    engine = HFAEngine()
    game_data = {
        "park": {"name": "Yankee Stadium"},
        "travel_source_away": "live",
        "miles_traveled_away": 2500,
        "time_zones_crossed_away": 3,
    }
    lh, la, meta = engine.get_adjusted_lambdas(4.5, 4.0, game_data)

    assert meta["travel_source"] == "live"
    assert meta["travel_penalty"] > 0.0
    assert la < 4.0  # real penalty applied


def test_hfa_engine_defaults_travel_source_to_live_when_key_absent():
    """Backtest compatibility: backtest_and_retrain.py populates
    miles_traveled_away/time_zones_crossed_away via its own DUP-001 travel
    calc but never sets travel_source_away — the default must be 'live' so
    the backtest's real (non-fabricated) numbers keep applying exactly as
    before this fix, not silently going neutral."""
    engine = HFAEngine()
    game_data = {
        "park": {"name": "Yankee Stadium"},
        "miles_traveled_away": 2500,
        "time_zones_crossed_away": 3,
        # no "travel_source_away" key at all
    }
    _, la, meta = engine.get_adjusted_lambdas(4.5, 4.0, game_data)
    assert meta["travel_source"] == "live"
    assert meta["travel_penalty"] > 0.0
    assert la < 4.0


def test_get_travel_fatigue_no_longer_fabricates_1000_1():
    """Programmatic check that the fabricated fallback is gone from the
    source, not just untested."""
    src = (REPO_ROOT / "data_fetchers.py").read_text()
    fn_body = src.split("def get_travel_fatigue")[1].split("def get_team_days_rest")[0]
    assert "miles = 1000" not in fn_body
    assert "assume mid-range travel" not in fn_body
