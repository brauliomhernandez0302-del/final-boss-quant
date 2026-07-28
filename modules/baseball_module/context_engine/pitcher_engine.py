"""
PITCHER ENGINE — Starting-pitcher quality and game-context adjustments
======================================================================

Adjusts λ_home / λ_away based on the STARTING PITCHER's characteristics.
Bullpen adjustments are handled by a separate Bullpen Engine (not this file).

Responsibilities:
  Starter quality    — SIERA → xFIP → xERA → FIP → ERA composite
                       + Statcast xwOBA allowed, barrel%, and K%-BB% overlays
                       + Bayesian regression to mean based on innings pitched
  Recent form        — NEUTRALIZADO (paso 8): ninguna señal de forma sobrevive
                       al control por nivel; ver _adjust_pitcher_form
  Matchup history    — era_vs_opp shrunk toward season ERA at < 15 IP
  Platoon splits     — L/R WHIP split × opposing lineup handedness composition
  Fatigue            — smooth gradient over days-rest and last-start pitch count

Does NOT handle:
  Home field advantage           — HFA Engine
  Ballpark / park factor         — ParkWeatherEngine
  Team offense                   — True Talent Engine
  Team defense                   — Defensive Efficiency Engine
  Bullpen quality / workload     — Bullpen Engine (separate)
  Travel fatigue of position players — HFA Engine
"""

from typing import Dict, Any, Tuple
import logging
from config import (
    PITCHER_ENGINE_WEIGHTS, LEAGUE_AVG_ERA, LEAGUE_AVG_WHIP, LEAGUE_AVG_XWOBA,
    LEAGUE_AVG_LHB_PCT as _LG_LHB_PCT,
)


def _first_present(*values):
    """Primer valor no-None. `0.0` es un valor válido y se conserva.

    Duplicado a propósito de park_weather_engine en vez de importarlo: los dos
    engines son hermanos y no hay módulo compartido entre ellos, así que
    importar uno desde el otro crearía una dependencia lateral por cuatro
    líneas. Lo que NO puede duplicarse es la constante de liga — ésa vive en
    config.py, que es de donde venía el bug de las dos copias de 0.45.
    """
    for v in values:
        if v is not None:
            return float(v)
    return 0.0

logger = logging.getLogger(__name__)

# League averages for normalisation (2024/2025 combined)
_LG_ERA            = LEAGUE_AVG_ERA     # 4.15
_LG_WHIP           = LEAGUE_AVG_WHIP    # 1.30
_LG_K_PCT          = 0.220              # starter league-avg K%
_LG_BB_PCT         = 0.080              # starter league-avg BB%
_LG_K_BB           = _LG_K_PCT - _LG_BB_PCT   # 0.140
_LG_XWOBA_ALLOWED  = LEAGUE_AVG_XWOBA   # single source of truth: config.py (aligned with Bullpen/TTE)
_LG_BRL_PCT        = 8.0               # barrel% allowed, starter avg

# ── Platoon: constantes MEDIDAS, no asumidas (auditoría paso 7, 2026-07-28) ───
#
# Razón poblacional WHIP(vs zurdos) / WHIP(vs derechos), sobre abridores con al
# menos 30 IP en CADA split (temporada 2026, 174 abridores de dos semanas de
# cartelera). Es la mediana, no la media, porque la distribución tiene cola:
#
#     RHP  n=88  mediana 1.1440   (sufre ~14% más contra zurdos)
#     LHP  n=11  mediana 0.8515   (sufre ~15% más contra derechos)
#
# Existen porque el efecto platoon es POBLACIONAL: regresar hacia "sin split"
# (razón 1.0) sesgaría a todo abridor con poca muestra en la misma dirección,
# subestimando su platoon siempre. Este prior lo evita.
#
# ⚠️ El de LHP se apoya en n=11. Es evidencia fina y el signo importa más que el
# valor exacto — re-medir cuando haya una temporada completa.
_PLATOON_RATIO_POBLACIONAL = {"R": 1.1440, "L": 0.8515}

# Varianza de TALENTO platoon individual, o sea cuánta de la dispersión que se
# observa entre abridores es real y cuánta es error muestral. Descomposición
# analítica sobre 119 RHP: (H+BB) ~ Poisson(WHIP·IP), así que la varianza
# relativa del WHIP de un split es 1/(WHIP·IP) y la de la razón se compone.
#
#     varianza observada de la razón   0.10691
#     varianza esperada sólo por ruido 0.07530
#     → sólo el 30% de lo que se ve es talento; el 70% es tamaño de muestra.
#
# Se usa como Var_talento en w = Var_talento/(Var_talento + Var_muestral_i), que
# da un peso POR ABRIDOR según su propia muestra en vez de un k global.
#
# Por qué no se barrió empíricamente como NB_DISPERSION: `backtest_and_retrain.py`
# no ejercita el camino de platoon en absoluto (cero referencias), así que no hay
# instrumento de Brier que llegue hasta acá. Ésta es la medición autocontenida
# que sí se puede hacer.
_PLATOON_VAR_TALENTO = 0.0316

# Nota sobre la métrica: se evaluó cambiar WHIP por (K-BB)/9 u OPS, que serían
# más independientes de la defensa. Medido con el mismo método, la fracción de
# talento individual es 30% para WHIP, 0% para (K-BB)/9 como diferencia
# (obs/ruido 0.93) y 0% para K/9 (0.92). En esta muestra WHIP es la única con
# señal individual detectable; el efecto de (K-BB) existe pero es puramente
# poblacional (media -0.911 vs zurdos), y eso ya lo captura el prior de arriba.
# n=119 y una temporada parcial: re-visitar con más datos.

# Bayesian stabilisation constants for ERA estimators (TBF at 50% reliability),
# branched by which estimator actually won the fallback chain below. A single
# uniform k=350 previously applied to all of them regardless of which one was
# selected — since SIERA is picked FIRST when available (fastest-stabilizing
# per its own cited research, k~250) but was assigned the SLOWEST group's
# constant, SIERA-based estimates were systematically under-trusted (at
# TBF=250, shrink_w came out ~58% instead of the correct 50%). Plain ERA
# (least-preferred fallback, most BABIP/defense/luck-contaminated) gets a
# larger k than FIP/xFIP — it deserves MORE shrinkage, not the same amount.
_K_TBF_SIERA = 250
_K_TBF_XFIP  = 350
_K_TBF_XERA  = 350
_K_TBF_FIP   = 350
_K_TBF_ERA_RAW = 450

# Away pitchers allow ~0.15 more ERA when pitching away from home.
# Source: documented across multiple baseball research papers (e.g., Baseball Prospectus, FG).
# Applied as a context adjustment on top of the Bayesian-regressed quality estimate.
_AWAY_ERA_PENALTY = 0.15


class PitcherEngine:
    """
    Adjusts Poisson λ values for the starting pitcher matchup.
    Each sub-factor returns a multiplier around 1.0; factors are combined
    via the delta formula: 1 + Σ((factor − 1) × weight).
    """

    def __init__(self):
        self.name    = "PitcherEngine"
        self.weights = dict(PITCHER_ENGINE_WEIGHTS)

    def adjust_for_pitchers(
        self,
        lh: float,
        la: float,
        game_data: Dict[str, Any],
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Adjusts λ_home and λ_away for the starter matchup.

        Away pitcher faces the home lineup  → adjusts λ_home.
        Home pitcher faces the away lineup  → adjusts λ_away.

        Returns (lh_adjusted, la_adjusted, metadata).
        """
        logger.info("Pitcher Engine — adjusting lambdas")
        logger.info("   Input: λ_h=%.3f  λ_a=%.3f", lh, la)

        pitcher_home = game_data.get("pitcher_home") or {}
        pitcher_away = game_data.get("pitcher_away") or {}

        # Away starter holds down the home lineup
        adj_away = self._calculate_pitcher_adjustment(pitcher_away, game_data, is_home=False)
        lh_new   = lh * adj_away["total_multiplier"]

        # Home starter holds down the away lineup
        adj_home = self._calculate_pitcher_adjustment(pitcher_home, game_data, is_home=True)
        la_new   = la * adj_home["total_multiplier"]

        logger.info(
            "   Away starter (%s): mult=%.3f  λ_home %.3f → %.3f",
            pitcher_away.get("name", "?"), adj_away["total_multiplier"], lh, lh_new,
        )
        logger.info(
            "   Home starter (%s): mult=%.3f  λ_away %.3f → %.3f",
            pitcher_home.get("name", "?"), adj_home["total_multiplier"], la, la_new,
        )

        metadata = {
            "pitcher_home": adj_home,
            "pitcher_away": adj_away,
        }
        return lh_new, la_new, metadata

    # ── Factor dispatch ────────────────────────────────────────────────────────

    def _calculate_pitcher_adjustment(
        self,
        pitcher: Dict[str, Any],
        game_data: Dict[str, Any],
        is_home: bool,
    ) -> Dict[str, Any]:
        quality  = self._adjust_pitcher_quality(pitcher, is_home=is_home)
        form     = self._adjust_pitcher_form(pitcher)
        matchup  = self._adjust_pitcher_matchup(pitcher, game_data, is_home)
        platoon  = self._adjust_pitcher_platoon(pitcher, game_data, is_home)
        fatigue  = self._adjust_pitcher_fatigue(pitcher)

        # Delta-weighted combination: preserves the direction and magnitude of each
        # sub-factor while mixing them proportionally by their assigned weights.
        total = 1.0 + (
            (quality - 1.0) * self.weights["pitcher_quality"] +
            (form    - 1.0) * self.weights["pitcher_form"]    +
            (matchup - 1.0) * self.weights["pitcher_matchup"] +
            (platoon - 1.0) * self.weights["pitcher_platoon"] +
            (fatigue - 1.0) * self.weights["pitcher_fatigue"]
        )

        return {
            "pitcher_name":     pitcher.get("name", "Unknown"),
            "quality_mult":     quality,
            "form_mult":        form,
            "matchup_mult":     matchup,
            "platoon_mult":     platoon,
            "fatigue_mult":     fatigue,
            "total_multiplier": max(0.65, min(1.45, total)),
        }

    # ── Quality ───────────────────────────────────────────────────────────────

    def _adjust_pitcher_quality(self, pitcher: Dict, is_home: bool = True) -> float:
        """
        Starter quality multiplier combining four orthogonal signals:

        1. ERA estimator (SIERA → xFIP → xERA → FIP → ERA)
           Bayesian-regressed toward league average based on IP (TBF proxy),
           so small samples early in the season don't overfit. Stabilization
           constant is branched by which estimator won (SIERA k=250,
           xFIP/xERA/FIP k=350, raw ERA k=450) rather than one uniform value.

        2. Statcast xwOBA allowed — expected batting value vs this pitcher
           per PA (removes BABIP luck from batted-ball outcomes).

        3. Statcast barrel% allowed — predicts HR/XBH better than HR/9.

        4. K%-BB% differential — only applied when the primary estimator
           is raw ERA (which doesn't encode K%/BB% itself). When SIERA/xFIP/
           xERA/FIP wins the fallback chain, this is skipped (set to 1.0):
           those estimators' own published formulas already substantially
           price in K%/BB%, confirmed empirically (corr(K%-BB%, SIERA)=-0.947
           on real 2025 data) — applying it on top double-counted command skill.

        Output clamped to [0.60, 1.45] (widened 2026-07-05 from [0.70,1.35],
        which was binding on ~11.5% of real qualified starters).
        """
        _era  = pitcher.get("era")
        era   = float(_era if _era is not None else _LG_ERA)
        fip   = pitcher.get("fip")
        xfip  = pitcher.get("xfip")
        xera  = pitcher.get("xera")
        siera = pitcher.get("siera")

        # Best available ERA estimator (most predictive → least), paired with
        # its own stabilization constant so the shrinkage matches the metric
        # actually selected instead of one uniform value for all of them.
        # `encodes_k_bb` tracks whether the selected estimator's own published
        # formula already prices in K%/BB% (SIERA, xFIP, xERA, and FIP are all
        # substantially built from exactly those inputs) — used below to gate
        # kbb_mult so it isn't re-pricing the same signal a second time.
        if siera is not None:
            primary, k_tbf, encodes_k_bb = float(siera), _K_TBF_SIERA, True
        elif xfip is not None:
            primary, k_tbf, encodes_k_bb = float(xfip), _K_TBF_XFIP, True
        elif xera is not None:
            primary, k_tbf, encodes_k_bb = float(xera), _K_TBF_XERA, True
        elif fip is not None:
            primary, k_tbf, encodes_k_bb = float(fip), _K_TBF_FIP, True
        else:
            primary, k_tbf, encodes_k_bb = era, _K_TBF_ERA_RAW, False

        # Bayesian regression: regress primary toward league avg based on IP sample.
        # At IP=0 → 100% league avg. At IP=k_tbf → 50/50. At IP=∞ → raw value.
        # `ip_mlb_equivalent`, no `innings_pitched`: esta n decide cuánto se le
        # cree al ERA propio frente a la media de liga, y un inning de Doble-A
        # no compra la misma credibilidad que uno de mayores. Con el crudo, un
        # abridor de AA con 150 innings quedaba prácticamente sin regresar.
        # Fallback al crudo cuando el campo no está (camino del backtest, que
        # arma sus dicts desde las cachés PIT y no pasa por la jerarquía de
        # respaldo de data_fetchers): ahí el comportamiento queda idéntico.
        _ip_eq   = pitcher.get("ip_mlb_equivalent")
        ip_cur   = float((_ip_eq if _ip_eq is not None else pitcher.get("innings_pitched")) or 0)
        tbf_est  = max(0.0, ip_cur * 4.3)   # ~4.3 TBF per IP for starters
        shrink_w = k_tbf / (k_tbf + tbf_est)
        primary_reg = primary * (1.0 - shrink_w) + _LG_ERA * shrink_w
        if not is_home:
            primary_reg += _AWAY_ERA_PENALTY
        skill_mult  = primary_reg / _LG_ERA

        # xwOBA allowed overlay — each 0.010 above avg ≈ +3% runs allowed
        est_woba = pitcher.get("est_woba")
        woba_mult = (
            1.0 + (float(est_woba) - _LG_XWOBA_ALLOWED) * 3.0
            if est_woba is not None else 1.0
        )

        # Barrel% allowed overlay — each 1 ppt above avg ≈ +1.2% runs allowed
        brl_pct = pitcher.get("brl_percent")
        brl_mult = (
            1.0 + max(0.0, float(brl_pct) - _LG_BRL_PCT) * 0.012
            if brl_pct is not None else 1.0
        )

        # K%-BB% overlay — elite command (high K, low BB) means fewer runs.
        # Only applied when the primary estimator did NOT already encode K%/BB%
        # (i.e. only for the raw-ERA fallback): empirically confirmed 2026-07-05
        # via real 2025 qualified-starter data that corr(K%-BB%, SIERA)=-0.947 and
        # corr(K%-BB%, FIP)=-0.808 — applying this on top of a SIERA/xFIP/xERA/FIP
        # skill_mult re-prices command skill that's already priced in, compounding
        # credit for one real signal. Raw ERA doesn't encode K/BB directly, so for
        # that thin-data fallback population this remains the only command signal
        # in the whole quality factor — kept at full strength there.
        # Each 1% above league avg K-BB (14%) → ~1.5% fewer runs allowed.
        k_pct  = pitcher.get("k_pct")
        bb_pct = pitcher.get("bb_pct")
        if not encodes_k_bb and k_pct is not None and bb_pct is not None:
            k_bb_diff = (float(k_pct) - float(bb_pct)) - _LG_K_BB
            kbb_mult  = max(0.88, min(1.12, 1.0 - k_bb_diff * 1.5))
        else:
            kbb_mult = 1.0

        # Bounds widened 2026-07-05 from [0.70,1.35]: that range was catching
        # real Cy Young-tier and back-of-rotation arms, not just garbage/missing
        # data — 11.5% of 52 real 2025 qualified starters (with full Savant
        # xwOBA/barrel overlays active) hit one boundary or the other. The
        # "hedge against single-game variance" argument for a tight clamp is a
        # category error: game-to-game outcome noise is already modeled by the
        # Monte Carlo's own dispersion parameter — compressing the MEAN quality
        # estimate here double-counts that uncertainty while destroying real
        # signal. New bounds set from the true unclamped distribution across
        # those 52 pitchers (range 0.6065-1.4067) with a small outward margin,
        # since a single season's qualified-starter sample likely doesn't
        # capture the true population's extremes.
        multiplier = max(0.60, min(1.45, skill_mult * woba_mult * brl_mult * kbb_mult))
        logger.debug(
            "   quality: prim=%.2f→%.2f(reg)  xwOBA=%.3f  brl=%.1f  kbb_mult=%.3f  → %.3f",
            primary, primary_reg,
            float(est_woba) if est_woba is not None else _LG_XWOBA_ALLOWED,
            float(brl_pct)  if brl_pct  is not None else _LG_BRL_PCT,
            kbb_mult, multiplier,
        )
        return multiplier

    # ── Form ──────────────────────────────────────────────────────────────────

    def _adjust_pitcher_form(self, pitcher: Dict) -> float:
        """NEUTRALIZADO — la "forma reciente" no predice nada, y como estaba
        codificada predecía al revés. Auditoría paso 8, 2026-07-28.

        Qué hacía: multiplicaba tres señales —nivel de ERA de los últimos 5
        arranques contra la de temporada, pendiente de la ERA, y % de quality
        starts— recortado a [0.85, 1.15]. Pesaba 0.256 —el segundo factor del motor,
        que en esta auditoría veníamos redondeando mal a "26%"—.

        Las tres salían de la MISMA lista de ~5 arranques (ver
        `data_fetchers.get_pitcher_game_log`), así que multiplicarlas contaba un
        solo dato tres veces — la clase de doble conteo que el gate
        `encodes_k_bb` evita cien líneas más arriba.

        LA EVIDENCIA. 1709 pares (arranque previo → siguiente) de 148 abridores
        reales, temporada 2026. Regresión con la ERA acumulada como CONTROL y
        errores estándar agrupados por pitcher, sobre la ERA cruda del siguiente
        arranque:

            ERA acumulada (control)   +0.4080   t = +2.25   ← lo único con señal
            ERA últimos 5             -0.0062   t = -0.04
            tendencia                 +0.0042   t = +0.06
            quality start %           +0.4601   t = +0.74

        Ninguna señal de forma sobrevive al control por nivel. Y sin ese control
        el factor completo daba r = -0.0226 contra el residual: no es que midiera
        poco, medía AL REVÉS. Lo que las señales captan es regresión a la media, y
        el motor las leía como persistencia.

        DOS ESPECIFICACIONES INTERMEDIAS DIERON FALSOS POSITIVOS, y quedan
        anotadas porque el error es reutilizable:
        - Con objetivo = residual (siguiente menos ERA acumulada), QS% daba
          t=+4.79. Artefacto: un QS% alto BAJA la ERA acumulada, que es el
          sustraendo, así que el residual sube por construcción.
        - Con errores iid en vez de agrupados, los t se inflan ~5x. Los arranques
          de un mismo pitcher no son observaciones independientes.

        POR QUÉ NEUTRALIZAR Y NO CORREGIR EL SIGNO: no hay nada que corregir. Con
        la especificación correcta ningún componente es distinto de cero. Invertir
        signos sería ajustar ruido.

        POR QUÉ NO SE REDISTRIBUYE ESE 0.256: la combinación es aditiva sobre deltas
        (`total = 1 + Σ wᵢ·(fᵢ-1)`), así que un factor neutro aporta exactamente
        cero sin importar su peso. No queda ningún hueco que rellenar; mover ese
        peso a los otros factores los AMPLIFICARÍA, y no hay evidencia de que
        estén sub-pesados. Sería un cambio distinto, con su propia carga de prueba.

        ALCANCE REAL DEL CAMBIO: el backtest no puede medir esto. En modo PIT
        llegan `era`, `era_last_5`, `era_trend` y `quality_start_pct` todas en
        None, así que el factor ya valía 1.0 en los 4.825 juegos — verificado. Es
        decir que esta corrección es SOLO-VIVO y su validación es la medición de
        arriba, no el Brier. La corrida de backtest sirve para confirmar que no
        movió nada, no para justificarlo.

        Se conserva la función (en vez de sacar el término de la suma) para que
        esta evidencia quede en el sitio donde alguien iría a buscarla antes de
        reinstaurar un ajuste de forma. Si se reinstaura, que sea con una señal
        que sobreviva a control por nivel y a errores agrupados.
        """
        return 1.0

    # ── Matchup ───────────────────────────────────────────────────────────────

    def _adjust_pitcher_matchup(
        self,
        pitcher: Dict,
        game_data: Dict,
        is_home: bool,
    ) -> float:
        """
        Historical ERA vs today's opponent, shrunk toward season ERA
        when sample is < 15 IP.
        """
        era_vs_team = pitcher.get("era_vs_opp")
        if era_vs_team is None:
            return 1.0

        _se         = pitcher.get("era")
        season_era  = float(_se if _se is not None else _LG_ERA)
        _ip         = pitcher.get("ip_vs_opp")
        ip_vs_opp   = float(_ip if _ip is not None else 0)

        if ip_vs_opp < 15:
            w = ip_vs_opp / 15.0
            era_vs_team = float(era_vs_team) * w + season_era * (1.0 - w)

        diff = float(era_vs_team) - season_era
        return max(0.85, min(1.15, 1.0 + diff * 0.06))

    # ── Platoon ───────────────────────────────────────────────────────────────

    def _adjust_pitcher_platoon(
        self,
        pitcher: Dict,
        game_data: Dict,
        is_home: bool,
    ) -> float:
        """
        WHIP-based platoon mismatch: lineup-composition-weighted WHIP vs overall WHIP.
        Home pitcher faces away lineup (away_lineup_lhb_pct) and vice versa.
        Returns 1.0 when split data is insufficient.
        """
        platoon = pitcher.get("platoon_splits")
        if not platoon:
            return 1.0

        vs_lhb = platoon.get("vs_lhb") or {}
        vs_rhb = platoon.get("vs_rhb") or {}
        if not vs_lhb or not vs_rhb:
            return 1.0

        whip_l_obs = float(vs_lhb.get("whip") or 0)
        whip_r_obs = float(vs_rhb.get("whip") or 0)
        ip_l = float(vs_lhb.get("ip") or 0)
        ip_r = float(vs_rhb.get("ip") or 0)
        whip_gen = float(pitcher.get("whip") or 0)
        if min(whip_l_obs, whip_r_obs, ip_l, ip_r, whip_gen) <= 0:
            return 1.0

        # ── Regresión del split (auditoría paso 7) ────────────────────────────
        # Antes esto entraba crudo, con un piso de 5 IP por split y nada más. El
        # resultado medido sobre 50 abridores: los de muestra chica se iban al
        # tope del recorte (8.2 IP → 0.930; 11.0 IP → 1.070) y los de muestra
        # grande daban ~1.00 (52-57 IP → 0.979-1.005). O sea que la magnitud del
        # factor la mandaba el error muestral, no la habilidad — y el recorte no
        # protegía del dato malo, era DONDE ATERRIZABA el dato malo.
        #
        # El prior NO es "este pitcher no tiene split": el efecto platoon es
        # poblacional y grande, así que regresar hacia diferencial cero sesgaría
        # a todos en la misma dirección. Se regresa hacia el split TÍPICO DE SU
        # MANO, anclado en su propio WHIP general.
        mano = pitcher.get("throws")
        ratio_pop = _PLATOON_RATIO_POBLACIONAL.get(mano)
        if ratio_pop is None:
            # Sin saber con qué mano lanza no hay prior poblacional que aplicar;
            # el neutro honesto es "no tiene split", no inventarle uno.
            ratio_pop = 1.0
        # Se encoge LA RAZÓN, no cada WHIP por separado. Es la razón la que tiene
        # la descomposición de varianza que da el peso, así que encoger los
        # componentes aplicaría un peso derivado para otra cantidad (difieren
        # hasta 0.032 en la razón resultante — chico, pero injustificado).
        #
        # Peso del dato propio = Var_talento / (Var_talento + Var_muestral_suya).
        # Ambos términos medidos, ninguno asumido: ver _PLATOON_VAR_TALENTO.
        ratio_obs = whip_l_obs / whip_r_obs
        var_muestral = ratio_obs ** 2 * (
            1.0 / (whip_l_obs * ip_l) + 1.0 / (whip_r_obs * ip_r)
        )
        w = _PLATOON_VAR_TALENTO / (_PLATOON_VAR_TALENTO + var_muestral)
        ratio = w * ratio_obs + (1.0 - w) * ratio_pop

        # Se reconstruyen los dos splits desde la razón encogida, anclados en su
        # WHIP GENERAL y en su exposición real a cada mano. Esto garantiza por
        # construcción que el promedio ponderado por exposición vuelva a dar su
        # WHIP general — o sea que el factor REDISTRIBUYE y no mueve el nivel.
        # El nivel ya lo cobra quality_mult (32%); que se filtrara acá sería
        # cobrarlo dos veces. Consecuencia comprobable: si la alineación que
        # enfrenta tiene la misma mezcla de manos que su exposición de
        # temporada, el factor da exactamente 1.0.
        expo_l = ip_l / (ip_l + ip_r)
        whip_r_aj = whip_gen / (expo_l * ratio + (1.0 - expo_l))
        vs_lhb = {"whip": ratio * whip_r_aj}
        vs_rhb = {"whip": whip_r_aj}

        # Misma cadena que park_weather_engine: lineup confirmado → mejor
        # estimación disponible (mezcla de alineación previa y roster) → media de
        # liga. Antes leía SÓLO la clave del lineup, así que en las corridas sin
        # lineup —el 83% de ellas, medido— usaba una constante teniendo el dato
        # del equipo a mano. Error medio de esa constante contra el roster real:
        # 0.076, casi la desviación completa entre equipos (0.085).
        _pre = "away" if is_home else "home"
        opp_lhb_pct = float(_first_present(
            game_data.get(f"{_pre}_lineup_lhb_pct"),
            game_data.get(f"{_pre}_lhb_pct"),
            _LG_LHB_PCT,
        ))
        opp_rhb_pct = 1.0 - opp_lhb_pct

        lhb_whip    = float(vs_lhb.get("whip", _LG_WHIP))
        rhb_whip    = float(vs_rhb.get("whip", _LG_WHIP))
        lineup_whip = opp_lhb_pct * lhb_whip + opp_rhb_pct * rhb_whip

        overall_whip = float(pitcher.get("whip", _LG_WHIP))
        if overall_whip <= 0:
            return 1.0

        return max(0.93, min(1.07, lineup_whip / overall_whip))

    # ── Fatigue ───────────────────────────────────────────────────────────────

    def _adjust_pitcher_fatigue(self, pitcher: Dict) -> float:
        """
        Smooth fatigue gradient from days rest and last-start pitch count.

        Days rest (optimal window: 4–5 days):
          0 d  → +9.0%  (back-to-back, extremely rare for starters)
          1 d  → +7.5%
          2 d  → +5.0%
          3 d  → +2.5%
          4–5 d → neutral
          6 d  → +0.8%  (rust begins)
          7 d  → +1.6%
          8+ d → +2.4% cap

        Pitch count (above 100 in last start):
          +0.12% per pitch above 100; capped at +4.8% (at 140 pitches).
        """
        _dr       = pitcher.get("days_rest")
        days_rest = int(_dr if _dr is not None else 4)

        if days_rest == 0:
            rest_adj = 1.090
        elif days_rest <= 3:
            rest_adj = 1.0 + (4 - days_rest) * 0.025
        elif days_rest >= 6:
            rest_adj = 1.0 + min((days_rest - 5) * 0.008, 0.024)
        else:
            rest_adj = 1.0   # 4–5 days: optimal

        _lpc     = pitcher.get("last_pitch_count")
        last_pc  = float(_lpc if _lpc is not None else 90)
        pc_mult  = (
            1.0 + min((last_pc - 100.0) * 0.0012, 0.048)
            if last_pc > 100 else 1.0
        )

        return max(0.95, min(1.12, rest_adj * pc_mult))


# ── Module-level helper ────────────────────────────────────────────────────────

def adjust_for_pitchers(
    lh: float,
    la: float,
    game_data: Dict[str, Any],
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Convenience wrapper used by run_module.py:
        from context_engine.pitcher_engine import adjust_for_pitchers
        lh, la, meta = adjust_for_pitchers(lh, la, game_data)
    """
    return PitcherEngine().adjust_for_pitchers(lh, la, game_data)
