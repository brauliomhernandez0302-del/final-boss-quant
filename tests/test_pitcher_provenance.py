"""Paso 5 — cuánto se le cree a un abridor según DE DÓNDE salió su dato.

`get_pitcher_stats_full_fallback` tiene 5 niveles de respaldo (MLB actual → MLB
anterior → AAA → AA → ERA del staff) y marcaba la procedencia con `is_fallback`
/ `fallback_tier`. Auditado el 2026-07-27: **nadie los leía**. Los dos lugares
donde la procedencia importa consumían `innings_pitched` crudo, que responde
"¿cuántos innings lanzó?" cuando la pregunta real es "¿cuánto le creo?":

  1. `pitcher_engine._adjust_pitcher_quality` lo usa como n de una regresión
     bayesiana hacia la media de liga. Con el crudo, un abridor de Doble-A con
     150 innings quedaba prácticamente sin regresar — su ERA de AA se trataba
     como ERA de mayores, y eso mueve λ. `quality_mult` es el 32% del motor.
  2. `compute_data_quality_confidence` lo pesa al 40%, su factor más grande, y
     su docstring dice explícitamente "innings pitched THIS SEASON".

Medido antes del cambio, los cuatro daban 0.65 exacto de confianza: MLB actual
con 30 IP, MLB del año anterior con 180, AAA con 100 y AA con 150. El score
medía el TAMAÑO de la muestra, nunca su procedencia.
"""
import pytest

import data_fetchers as df
from core.value_detector import compute_data_quality_confidence as confianza


# ── el descuento en el origen ─────────────────────────────────────────────────

def test_los_factores_estan_ordenados_y_acotados():
    """Un inning vale más cuanto más se parece a MLB de esta temporada."""
    assert 0 < df._IP_EQ_AA < df._IP_EQ_AAA < df._IP_EQ_MLB_PREV_SEASON < 1.0


def test_ninguna_fuente_de_respaldo_vale_cero():
    """180 innings del año pasado dicen algo: tirarlos sería tan deshonesto
    como creerles enteros. El único cero legítimo es el ERA del staff, que no
    es una muestra de este abridor en absoluto."""
    assert df._IP_EQ_MLB_PREV_SEASON > 0
    assert df._IP_EQ_AAA > 0
    assert df._IP_EQ_AA > 0


@pytest.mark.parametrize("tier,ip,factor", [
    ("mlb_prev_season", 180.0, df._IP_EQ_MLB_PREV_SEASON),
    ("aaa_current",     100.0, df._IP_EQ_AAA),
    ("aa_current",      150.0, df._IP_EQ_AA),
])
def test_cada_tier_descuenta_lo_suyo(monkeypatch, tier, ip, factor):
    api = df.MLBStatsAPI()
    stats = {"era": 3.00, "whip": 1.10, "fip": 3.10, "innings_pitched": ip}

    monkeypatch.setattr(api, "get_pitcher_stats_with_fallback", lambda *a, **k: (None, None))
    monkeypatch.setattr(api, "get_pitcher_stats",
                        lambda pid, season=None: dict(stats) if tier == "mlb_prev_season" else None)
    monkeypatch.setattr(
        api, "_fetch_milb_stats",
        lambda pid, s, sport_id: dict(stats) if (
            (sport_id == 11 and tier == "aaa_current") or (sport_id == 12 and tier == "aa_current")
        ) else None,
    )

    out, src = api.get_pitcher_stats_full_fallback(1, 2026)
    assert src == tier
    assert out["ip_mlb_equivalent"] == pytest.approx(ip * factor)
    assert out["innings_pitched"] == ip, "el crudo no se toca: responde otra pregunta"


def test_mlb_de_esta_temporada_no_se_descuenta(monkeypatch):
    api = df.MLBStatsAPI()
    monkeypatch.setattr(api, "get_pitcher_stats_with_fallback",
                        lambda *a, **k: ({"era": 3.0, "innings_pitched": 120.0}, "home_split"))
    out, src = api.get_pitcher_stats_full_fallback(1, 2026)
    assert out["ip_mlb_equivalent"] == 120.0


def test_el_era_del_staff_vale_cero(monkeypatch):
    """No es una muestra de ESTE abridor: es el promedio de su cuerpo de
    lanzadores. Cero información sobre quién abre."""
    api = df.MLBStatsAPI()
    monkeypatch.setattr(api, "get_pitcher_stats_with_fallback", lambda *a, **k: (None, None))
    monkeypatch.setattr(api, "get_pitcher_stats", lambda *a, **k: None)
    monkeypatch.setattr(api, "_fetch_milb_stats", lambda *a, **k: None)
    out, src = api.get_pitcher_stats_full_fallback(1, 2026, team_pitching={"team_era": 4.1})
    assert src == "team_staff_era"
    assert out["ip_mlb_equivalent"] == 0.0


# ── consumidor 1: la regresión del motor ──────────────────────────────────────

def _quality(ip_crudo, ip_eq=None):
    import sys
    sys.path.insert(0, "modules/baseball_module")
    from context_engine.pitcher_engine import PitcherEngine
    p = {"era": 3.00, "innings_pitched": ip_crudo}
    if ip_eq is not None:
        p["ip_mlb_equivalent"] = ip_eq
    return PitcherEngine()._adjust_pitcher_quality(p, is_home=True)


def test_doble_a_se_regresa_mucho_mas_que_mlb():
    """Mismo ERA, mismos innings crudos, distinta liga."""
    mlb = _quality(150, 150.0)
    aa = _quality(150, 150.0 * df._IP_EQ_AA)
    assert aa > mlb, "el de AA debe quedar MÁS cerca de la media de liga"
    assert aa - mlb > 0.05, f"el efecto tiene que ser material, no cosmético (Δ={aa-mlb:.4f})"


def test_sin_el_campo_el_comportamiento_es_identico():
    """El backtest arma sus dicts desde las cachés PIT y no pasa por la
    jerarquía de respaldo, así que nunca trae `ip_mlb_equivalent`. Ahí el
    resultado tiene que quedar exactamente igual que antes de este cambio."""
    assert _quality(150, None) == _quality(150, 150.0)


def test_mlb_actual_no_se_mueve():
    """El fix es inerte donde el dato ya era genuino."""
    assert _quality(150, 150.0) == pytest.approx(_quality(150, None))


# ── consumidor 2: la confianza ────────────────────────────────────────────────

def _conf(sp_ip):
    return confianza({
        "home_sp_ip": sp_ip, "away_sp_ip": 999,
        "home_prior_weight": 0, "away_prior_weight": 0,
        "home_kalman_n_obs": 99, "away_kalman_n_obs": 99,
    })


def test_la_confianza_ya_distingue_procedencia():
    """Antes: AAA-100IP, AA-150IP y MLB-30IP daban 0.65 los tres."""
    mlb = _conf(30.0)
    aaa = _conf(100.0 * df._IP_EQ_AAA)
    aa = _conf(150.0 * df._IP_EQ_AA)
    staff = _conf(0.0)
    assert mlb > aaa > aa > staff, f"orden roto: {mlb} {aaa} {aa} {staff}"


def test_la_saturacion_a_30_ip_sigue_ahi_y_es_deliberada():
    """Límite conocido, documentado a propósito: q_sp satura en ip/30, así que
    una muestra grande del año anterior (180 IP → 90 equivalentes) sigue dando
    crédito pleno EN LA CONFIANZA. El descuento sí muerde entero en la regresión
    del motor, que no satura. No es un olvido — es dónde llega este fix."""
    assert _conf(180.0 * df._IP_EQ_MLB_PREV_SEASON) == _conf(180.0)
    assert _conf(29.0) < _conf(30.0)
