"""Paso 7 — el ajuste platoon: que la magnitud la mande la habilidad y no el ruido.

Dos defectos, auditados el 2026-07-28.

(A) La notación de innings del béisbol no se convertía. "5.2" son 5⅔, pero se
    hacía `float("5.2")=5.2`. `_ip_to_float` existía en la misma clase, con
    docstring explícito, y la usaba UN solo sitio de nueve. Todos los demás eran
    denominadores de tasas, así que el sesgo es sistemático y de un solo signo.

    Lo grave era la asimetría: el WHIP y el ERA GENERALES vienen del campo ya
    calculado por la API, mientras que los splits, el FIP, K/9 y los IP por
    inicio se recalculaban localmente con el denominador mal. Cualquier RAZÓN
    entre ambos quedaba sesgada de un lado — que es justo lo que computa
    `_adjust_pitcher_platoon`. Medido: WHIP de split inflado en el 100% de 105
    casos, y en el camino de CALIDAD (32% del motor) FIP +0.0072 c/9 y K/9 +0.46%.

(B) El factor platoon no tenía ninguna regresión por tamaño de muestra — la
    única de las 5 sub-señales del motor sin ella. Medido sobre 50 abridores:
    los de muestra chica se iban al tope del recorte (8.2 IP → 0.930; 11 IP →
    1.070) y los de muestra grande daban ~1.00. La magnitud la mandaba el error
    muestral, y el recorte no protegía del dato malo: era donde aterrizaba.
"""
import pytest

import data_fetchers as df


# ── (A) la notación de innings ────────────────────────────────────────────────

@pytest.mark.parametrize("crudo,esperado", [
    ("5.0", 5.0),
    ("5.1", 5 + 1 / 3),      # un out
    ("5.2", 5 + 2 / 3),      # dos outs — float() daría 5.2
    ("0.1", 1 / 3),
    (12, 12.0),
])
def test_la_notacion_de_innings_se_convierte(crudo, esperado):
    assert df.MLBStatsAPI._ip_to_float(crudo) == pytest.approx(esperado)


def test_el_parseo_crudo_siempre_subestima():
    """Por eso el sesgo tenía un solo signo: denominador chico, tasa inflada."""
    for crudo in ("1.1", "1.2", "40.1", "40.2"):
        assert float(crudo) < df.MLBStatsAPI._ip_to_float(crudo)


def test_ningun_sitio_quedo_con_el_parseo_crudo():
    """Eran nueve y sólo uno usaba el conversor. El bug estaba en los otros ocho."""
    fuente = open(df.__file__.replace(".pyc", ".py"), encoding="utf-8").read()
    for linea in fuente.splitlines():
        if "inningsPitched" not in linea or linea.lstrip().startswith("#"):
            continue
        if "float(" in linea and "_ip_to_float" not in linea:
            pytest.fail(f"parseo crudo de innings sobrevivió: {linea.strip()}")


# ── (B) la regresión del split ────────────────────────────────────────────────

def _factor(ip, whip_l, whip_r, whip_gen=1.20, throws="R", lhb=0.40):
    import sys
    sys.path.insert(0, "modules/baseball_module")
    from context_engine.pitcher_engine import PitcherEngine
    return PitcherEngine()._adjust_pitcher_platoon(
        {"whip": whip_gen, "throws": throws, "platoon_splits": {
            "vs_lhb": {"whip": whip_l, "ip": ip}, "vs_rhb": {"whip": whip_r, "ip": ip}}},
        {"away_lhb_pct": lhb}, is_home=True,
    )


def _crudo(whip_l, whip_r, whip_gen=1.20, lhb=0.40):
    return max(0.93, min(1.07, (lhb * whip_l + (1 - lhb) * whip_r) / whip_gen))


def test_una_muestra_chica_casi_no_mueve_el_factor():
    """8 innings de split es ruido: el factor tiene que quedarse cerca del prior."""
    chico = _factor(8, whip_l=2.20, whip_r=1.10)
    assert abs(chico - 1.0) < abs(_crudo(2.20, 1.10) - 1.0), "tiene que encogerse"


def test_una_muestra_grande_conserva_su_señal():
    """Con 60 IP por lado el dato propio pesa mucho más que el prior."""
    chico = _factor(8, whip_l=2.20, whip_r=1.10)
    grande = _factor(60, whip_l=2.20, whip_r=1.10)
    assert abs(grande - 1.0) > abs(chico - 1.0)


def test_el_encogimiento_es_monotono_en_la_muestra():
    desvios = [abs(_factor(ip, 2.20, 1.10) - 1.0) for ip in (5, 15, 40, 80)]
    assert desvios == sorted(desvios), f"más muestra ⇒ más peso propio: {desvios}"


def test_el_recorte_deja_de_ser_donde_cae_el_ruido():
    """Antes, 7 de 51 abridores reales quedaban pegados al tope [0.93, 1.07] y
    eran justamente los de muestra chica. Con la regresión, 1 de 51."""
    for ip in (5, 8, 12):
        f = _factor(ip, whip_l=3.00, whip_r=0.60)   # split absurdo, muestra mínima
        assert 0.9301 < f < 1.0699, f"con {ip} IP no debería llegar al tope (dio {f})"


# ── el prior poblacional, que es lo que hace correcto el encogimiento ─────────

def test_derechos_y_zurdos_tienen_priors_opuestos():
    """Medido: RHP sufre ~14% más contra zurdos (razón 1.144), LHP al revés
    (0.852). Regresar hacia 'sin split' habría sesgado a todos igual."""
    from context_engine.pitcher_engine import _PLATOON_RATIO_POBLACIONAL as P
    assert P["R"] > 1.0 > P["L"]


def test_sin_saber_la_mano_no_se_inventa_un_split():
    """El neutro honesto es 'no tiene split', no asumirle el de un derecho."""
    r = _factor(10, 2.20, 1.10, throws="R")
    n = _factor(10, 2.20, 1.10, throws=None)
    assert r != n


def test_la_varianza_de_talento_es_la_medida():
    """30% de la dispersión observada es talento, 70% es muestra. De ahí sale el
    peso por abridor, en vez de un k global inventado."""
    from context_engine.pitcher_engine import _PLATOON_VAR_TALENTO
    assert _PLATOON_VAR_TALENTO == pytest.approx(0.0316)


def test_si_la_mezcla_del_lineup_iguala_su_exposicion_el_factor_es_neutro():
    """La invariante que hace correcto a este factor: sólo REDISTRIBUYE.

    El nivel del abridor ya lo cobra `quality_mult` (32% del motor). Si el ajuste
    platoon moviera además el nivel, sería cobrarlo dos veces. Reconstruyendo los
    splits desde la razón encogida y anclándolos en su WHIP general, el promedio
    ponderado por exposición vuelve a dar exactamente ese WHIP — así que cuando
    la alineación que enfrenta tiene la misma mezcla de manos que su exposición
    de temporada, no hay desajuste que cobrar y el factor es 1.0 exacto.

    Encogiendo cada WHIP por separado —la primera versión de este fix— esto NO
    se cumplía: el nivel se filtraba.
    """
    import sys
    sys.path.insert(0, "modules/baseball_module")
    from context_engine.pitcher_engine import PitcherEngine
    e = PitcherEngine()
    for ip_l, ip_r in [(30.0, 70.0), (50.0, 50.0), (20.0, 90.0)]:
        expo_l = ip_l / (ip_l + ip_r)
        f = e._adjust_pitcher_platoon(
            {"whip": 1.20, "throws": "R", "platoon_splits": {
                "vs_lhb": {"whip": 1.55, "ip": ip_l},
                "vs_rhb": {"whip": 1.05, "ip": ip_r}}},
            {"away_lhb_pct": expo_l}, is_home=True,
        )
        assert f == pytest.approx(1.0, abs=1e-9), (
            f"exposición {expo_l:.2f} ⇒ el factor debe ser neutro, dio {f}")


def test_una_alineacion_mas_zurda_castiga_a_un_derecho():
    """Y el signo tiene que ser el correcto, no sólo la magnitud."""
    import sys
    sys.path.insert(0, "modules/baseball_module")
    from context_engine.pitcher_engine import PitcherEngine
    e = PitcherEngine()
    p = {"whip": 1.20, "throws": "R", "platoon_splits": {
        "vs_lhb": {"whip": 1.55, "ip": 50.0}, "vs_rhb": {"whip": 1.05, "ip": 50.0}}}
    poco = e._adjust_pitcher_platoon(p, {"away_lhb_pct": 0.20}, is_home=True)
    mucho = e._adjust_pitcher_platoon(p, {"away_lhb_pct": 0.80}, is_home=True)
    assert mucho > 1.0 > poco, f"más zurdos ⇒ peor para el derecho ({poco}, {mucho})"


def test_datos_incompletos_devuelven_neutro():
    for kwargs in ({"whip_l": 0}, {"whip_r": 0}, {"whip_gen": 0}):
        base = dict(ip=40, whip_l=1.30, whip_r=1.10)
        base.update(kwargs)
        assert _factor(**base) == 1.0
