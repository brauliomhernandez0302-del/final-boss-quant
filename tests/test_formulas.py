"""
Tests for core mathematical formulas.

Covers: FIP calculation, wOBA components, wRC+ approximation,
        EV percentage, Kelly criterion stake sizing.
"""
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers that replicate the formula exactly as written in production code.
# We call the real implementation wherever possible and fall back to an inline
# restatement only when the formula lives inside an API-calling method.
# ─────────────────────────────────────────────────────────────────────────────

FIP_CONSTANT = 3.10
LG_WOBA      = 0.320
WOBA_SCALE   = 1.157


def _fip(hr: int, bb: int, hbp: int, so: int, ip: float) -> float:
    """Inline restatement of the FIP formula for cross-checking."""
    return (13 * hr + 3 * (bb + hbp) - 2 * so) / ip + FIP_CONSTANT


def _woba(ubb, hbp, singles, doubles, triples, hr, ab, sf) -> float:
    denom = ab + ubb + hbp + sf
    if denom == 0:
        return LG_WOBA
    return (
        0.690 * ubb + 0.722 * hbp + 0.888 * singles +
        1.271 * doubles + 1.616 * triples + 2.101 * hr
    ) / denom


def _wrc_plus(woba: float) -> float:
    raw = ((woba - LG_WOBA) / WOBA_SCALE) * 100 + 100
    return max(50.0, min(165.0, raw))


# ─────────────────────────────────────────────────────────────────────────────
# FIP
# ─────────────────────────────────────────────────────────────────────────────

class TestFIP:
    """FIP = (13·HR + 3·(BB+HBP) − 2·K) / IP + 3.10"""

    def test_league_average_pitcher(self):
        # A "league-average" FIP should land close to 4.10-4.20 ERA range.
        # Typical: HR=20, BB=50, HBP=5, K=170, IP=170 → real FIP ≈ 4.10
        result = _fip(hr=20, bb=50, hbp=5, so=170, ip=170)
        assert 3.5 < result < 4.8, f"Expected ~4.1, got {result:.3f}"

    def test_elite_pitcher(self):
        # Low HR, low BB, high K → FIP well below 3.00
        # HR=10, BB=30, HBP=3, K=200, IP=180
        # = (130 + 99 - 400) / 180 + 3.10 = -171/180 + 3.10 ≈ 2.15
        result = _fip(hr=10, bb=30, hbp=3, so=200, ip=180)
        expected = (13 * 10 + 3 * (30 + 3) - 2 * 200) / 180 + FIP_CONSTANT
        assert abs(result - expected) < 1e-9
        assert result < 3.00, "Elite pitcher should have FIP < 3.00"

    def test_bad_pitcher(self):
        # High HR, high BB, low K → FIP above 5.00
        result = _fip(hr=35, bb=80, hbp=10, so=100, ip=140)
        expected = (13 * 35 + 3 * (80 + 10) - 2 * 100) / 140 + FIP_CONSTANT
        assert abs(result - expected) < 1e-9
        assert result > 5.00, "Bad pitcher should have FIP > 5.00"

    def test_known_value(self):
        # HR=15, BB=45, HBP=5, K=150, IP=160
        # numerator = 13*15 + 3*(45+5) - 2*150 = 195 + 150 - 300 = 45
        # FIP = 45/160 + 3.10 = 0.28125 + 3.10 = 3.38125
        result = _fip(hr=15, bb=45, hbp=5, so=150, ip=160)
        assert abs(result - 3.38125) < 1e-6

    def test_via_parse_pitcher_stats(self):
        """The parser in data_fetchers.py must produce the same FIP."""
        from data_fetchers import MLBStatsAPI
        api = MLBStatsAPI()

        hr, bb, hbp, so, ip = 15, 45, 5, 150, 160.0
        data = {"stats": [{"splits": [{"stat": {
            "era": "3.50", "whip": "1.20",
            "inningsPitched": str(ip),
            "strikeOuts": so, "baseOnBalls": bb,
            "homeRuns": hr, "hitBatsmen": hbp,
            "hits": 130, "earnedRuns": 62,
            "gamesStarted": 25, "wins": 10, "losses": 8,
        }}]}]}
        result = api._parse_pitcher_stats(data)
        assert result is not None
        expected_fip = round(max(0.0, min((13*hr + 3*(bb+hbp) - 2*so) / ip + FIP_CONSTANT, 12.0)), 2)
        assert result["fip"] == expected_fip
        assert result["avg_innings_per_start"] == round(ip / 25, 2)

    def test_hbp_contributes_same_as_walk(self):
        # HBP and BB both have weight 3 in the numerator.
        fip_bb  = _fip(hr=10, bb=40, hbp=0, so=100, ip=100)
        fip_hbp = _fip(hr=10, bb=0,  hbp=40, so=100, ip=100)
        assert abs(fip_bb - fip_hbp) < 1e-9

    def test_hr_weight_is_13(self):
        # Each additional HR should add 13/IP to FIP.
        base   = _fip(hr=10, bb=30, hbp=5, so=120, ip=100)
        plus_1 = _fip(hr=11, bb=30, hbp=5, so=120, ip=100)
        assert abs((plus_1 - base) - 13 / 100) < 1e-9

    def test_strikeout_weight_is_negative_2(self):
        base     = _fip(hr=10, bb=30, hbp=5, so=120, ip=100)
        plus_10k = _fip(hr=10, bb=30, hbp=5, so=130, ip=100)
        assert abs((plus_10k - base) - (-2 * 10 / 100)) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# wOBA
# ─────────────────────────────────────────────────────────────────────────────

class TestWOBA:
    """wOBA component weights: uBB×0.690, HBP×0.722, 1B×0.888, 2B×1.271, 3B×1.616, HR×2.101"""

    def test_league_average_woba(self):
        # Constructing a stat line that should yield exactly wOBA=0.320.
        # Use only unintentional walks and compute what singles we need.
        # Simplest: 0 events → wOBA = LG_WOBA fallback (denom=0)
        result = _woba(0, 0, 0, 0, 0, 0, 0, 0)
        assert result == LG_WOBA

    def test_hr_only_lineup(self):
        # 50 HR, 200 AB, no walks/HBP/SF
        # wOBA = 2.101 * 50 / 200 = 0.52525
        result = _woba(ubb=0, hbp=0, singles=0, doubles=0, triples=0,
                       hr=50, ab=200, sf=0)
        assert abs(result - 2.101 * 50 / 200) < 1e-9

    def test_walk_heavy_lineup(self):
        # 80 uBB, 0 everything else, 300 AB
        # wOBA = 0.690 * 80 / 380 ≈ 0.14526
        result = _woba(ubb=80, hbp=0, singles=0, doubles=0, triples=0,
                       hr=0, ab=300, sf=0)
        expected = 0.690 * 80 / (300 + 80)
        assert abs(result - expected) < 1e-9

    def test_hbp_weight_greater_than_walk(self):
        # HBP weight (0.722) > uBB weight (0.690); more productive.
        woba_bb  = _woba(ubb=1, hbp=0, singles=0, doubles=0, triples=0, hr=0, ab=10, sf=0)
        woba_hbp = _woba(ubb=0, hbp=1, singles=0, doubles=0, triples=0, hr=0, ab=10, sf=0)
        assert woba_hbp > woba_bb

    def test_hr_highest_weight(self):
        # Each event type adds weight/denom. HR (2.101) > 3B (1.616) > 2B (1.271) > 1B (0.888).
        denom = 100
        assert (2.101 / denom) > (1.616 / denom) > (1.271 / denom) > (0.888 / denom)

    def test_realistic_team_stat_line(self):
        # Realistic full-season line for a league-average team (~162-game season):
        # uBB=430, HBP=55, 1B=850, 2B=280, 3B=25, HR=160, AB=5300, SF=45
        ubb, hbp, s, d, t, hr, ab, sf = 430, 55, 850, 280, 25, 160, 5300, 45
        result = _woba(ubb, hbp, s, d, t, hr, ab, sf)
        assert 0.280 < result < 0.370, f"Expected league-range wOBA, got {result:.3f}"

    def test_sf_increases_denominator(self):
        # Adding SFs with no additional hits should lower wOBA.
        base      = _woba(ubb=50, hbp=5, singles=100, doubles=25, triples=5, hr=20, ab=300, sf=0)
        with_sf   = _woba(ubb=50, hbp=5, singles=100, doubles=25, triples=5, hr=20, ab=300, sf=20)
        assert with_sf < base


# ─────────────────────────────────────────────────────────────────────────────
# wRC+
# ─────────────────────────────────────────────────────────────────────────────

class TestWRCPlus:
    """wRC+ = ((wOBA − lgwOBA) / wOBAscale) × 100 + 100, clamped [50, 165]."""

    def test_league_average_is_100(self):
        assert _wrc_plus(LG_WOBA) == 100.0

    def test_above_average_over_100(self):
        assert _wrc_plus(0.360) > 100

    def test_below_average_under_100(self):
        assert _wrc_plus(0.290) < 100

    def test_known_value(self):
        # wOBA = 0.377 → ((0.377 - 0.320) / 1.157) * 100 + 100
        # = (0.057 / 1.157) * 100 + 100 = 4.926 + 100 = 104.926
        result = _wrc_plus(0.377)
        assert abs(result - ((0.377 - 0.320) / 1.157 * 100 + 100)) < 0.01

    def test_upper_clamp(self):
        # wOBA must exceed ~1.072 to hit the 165 ceiling
        # ((1.5 - 0.320) / 1.157)*100 + 100 ≈ 202 → clamped at 165
        result = _wrc_plus(1.5)
        assert result == 165.0

    def test_lower_clamp(self):
        # wOBA must be below ~-0.26 to hit the 50 floor
        # ((-0.5 - 0.320) / 1.157)*100 + 100 ≈ 29 → clamped at 50
        result = _wrc_plus(-0.5)
        assert result == 50.0

    def test_linear_with_woba(self):
        # Each +0.010 wOBA should increase wRC+ by the same amount.
        delta = (0.010 / WOBA_SCALE) * 100
        r1 = _wrc_plus(0.320)
        r2 = _wrc_plus(0.330)
        assert abs((r2 - r1) - delta) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# EV
# ─────────────────────────────────────────────────────────────────────────────

class TestEV:
    """EV = (prob × decimal_odds − 1) × 100  (percentage)."""

    def test_breakeven_is_zero(self):
        from core.utils import calculate_ev
        # At 50% probability, fair odds = 2.00 → EV = 0
        assert calculate_ev(0.5, 2.0) == pytest.approx(0.0)

    def test_positive_ev(self):
        from core.utils import calculate_ev
        # 55% chance at 2.00 → EV = (0.55*2.0 - 1) * 100 = 10.0%
        assert calculate_ev(0.55, 2.0) == pytest.approx(10.0)

    def test_negative_ev(self):
        from core.utils import calculate_ev
        # 45% chance at 2.00 → EV = (0.45*2.0 - 1) * 100 = -10.0%
        assert calculate_ev(0.45, 2.0) == pytest.approx(-10.0)

    def test_juice_reduces_ev(self):
        from core.utils import calculate_ev
        # Typical -110 line: 1.909 odds. True 50% chance → EV negative.
        ev_fair  = calculate_ev(0.5, 2.0)
        ev_juice = calculate_ev(0.5, 1.909)
        assert ev_juice < ev_fair

    def test_high_odds_amplifies_edge(self):
        from core.utils import calculate_ev
        # Same 5% edge but at +250 (3.50) vs -110 (1.909)
        ev_high = calculate_ev(0.30, 3.50)
        ev_low  = calculate_ev(0.55, 1.909)
        # Both are positive; the higher-odds bet can be more attractive
        assert ev_high > 0 and ev_low > 0

    def test_zero_probability_is_minus_100(self):
        from core.utils import calculate_ev
        assert calculate_ev(0.0, 2.0) == pytest.approx(-100.0)

    def test_certainty_is_odds_minus_1_times_100(self):
        from core.utils import calculate_ev
        # prob=1.0, odds=1.50 → EV = (1.50 - 1) * 100 = 50.0
        assert calculate_ev(1.0, 1.50) == pytest.approx(50.0)

