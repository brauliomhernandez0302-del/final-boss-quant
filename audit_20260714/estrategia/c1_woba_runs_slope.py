"""C1 - La conversion wOBA->carreras del TTE (ratio contra LG_RPG) vs la pendiente
real observada (regresion lineal simple team xwOBA -> RS/G real). Solo lectura.

Compara tambien la dispersion cruzada de equipos: lambda del TTE (compuesto
completo, casi sin encogimiento a fin de temporada) contra el RS/G real.

Requiere data/pit_cache_merged.db (namespace savant.team_offense.rolling) y
data/predictions_history.db (game_outcomes, source='backtest').
"""
import sqlite3, json, sys
import numpy as np

sys.path.insert(0, '/home/raulio')
from modules.baseball_module.offense.tte_formula import regress, season_lambda
import config

ABBR2NAME = {
 'ATH': 'Athletics', 'ATL': 'Atlanta Braves', 'AZ': 'Arizona Diamondbacks', 'BAL': 'Baltimore Orioles',
 'BOS': 'Boston Red Sox', 'CHC': 'Chicago Cubs', 'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians',
 'COL': 'Colorado Rockies', 'CWS': 'Chicago White Sox', 'DET': 'Detroit Tigers', 'HOU': 'Houston Astros',
 'KC': 'Kansas City Royals', 'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers', 'MIA': 'Miami Marlins',
 'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins', 'NYM': 'New York Mets', 'NYY': 'New York Yankees',
 'PHI': 'Philadelphia Phillies', 'PIT': 'Pittsburgh Pirates', 'SD': 'San Diego Padres', 'SEA': 'Seattle Mariners',
 'SF': 'San Francisco Giants', 'STL': 'St. Louis Cardinals', 'TB': 'Tampa Bay Rays', 'TEX': 'Texas Rangers',
 'TOR': 'Toronto Blue Jays', 'WSH': 'Washington Nationals',
}
NAME_ALIASES = {"Athletics": ["Athletics", "Oakland Athletics"]}

LG_XWOBA = config.LEAGUE_AVG_XWOBA
LG_RPG = config.LEAGUE_AVG_RUNS
LG_BARREL_PA, LG_BB_PCT, LG_K_PCT = 0.088, 0.086, 0.224
LG_WOBA_SCALE = 1.157  # ya vive en true_talent_engine.py, hoy solo alimenta wrc_plus_approx

pit = sqlite3.connect('/home/raulio/data/pit_cache_merged.db')
pcur = pit.cursor()
gdb = sqlite3.connect('/home/raulio/data/predictions_history.db')
gcur = gdb.cursor()


def latest_row(abbr, season, as_of_date):
    pcur.execute(
        """select data_json from pit_metric_cache
           where namespace='savant.team_offense.rolling' and entity_id=? and season=? and as_of_date<=?
           order by as_of_date desc limit 1""",
        (abbr, season, as_of_date),
    )
    r = pcur.fetchone()
    return json.loads(r[0]) if r else None


for season, cutoff in [(2024, '2024-09-30'), (2025, '2025-09-30')]:
    xw, rsg, comp, pas = [], [], [], []
    for ab, name in ABBR2NAME.items():
        d = latest_row(ab, season, cutoff)
        if not d or d.get('pa', 0) <= 100:
            continue
        names = NAME_ALIASES.get(name, [name])
        rows_h, rows_a = [], []
        for nm in names:
            gcur.execute(
                "select actual_home_runs from game_outcomes where season=? and source='backtest' "
                "and home_team=? and actual_home_runs is not null", (season, nm))
            rows_h += [r[0] for r in gcur.fetchall()]
            gcur.execute(
                "select actual_away_runs from game_outcomes where season=? and source='backtest' "
                "and away_team=? and actual_away_runs is not null", (season, nm))
            rows_a += [r[0] for r in gcur.fetchall()]
        if not rows_h or not rows_a:
            continue
        pa, bip = d['pa'], d.get('bip', d.get('batted_ball_count', 0))
        xwoba_reg = regress(d['est_woba'], LG_XWOBA, pa, 150)
        barrel_rate = d['barrel_count'] / bip if bip > 0 else 0
        barrel_reg = regress(barrel_rate, LG_BARREL_PA, bip, 120)
        bb_reg = regress(d['bb_pct'], LG_BB_PCT, pa, 120)
        k_reg = regress(d['k_pct'], LG_K_PCT, pa, 60)
        lam, _ = season_lambda(xwoba_reg=xwoba_reg, barrel_reg=barrel_reg, bb_reg=bb_reg, k_reg=k_reg,
                                lg_xwoba=LG_XWOBA, lg_barrel_rate=LG_BARREL_PA,
                                lg_bb_pct=LG_BB_PCT, lg_k_pct=LG_K_PCT, lg_rpg=LG_RPG)
        xw.append(d['est_woba']); rsg.append((np.mean(rows_h) + np.mean(rows_a)) / 2)
        comp.append(lam); pas.append(pa)

    xw, rsg, comp, pas = map(np.array, (xw, rsg, comp, pas))
    slope_emp = np.polyfit(xw, rsg, 1)[0]
    slope_model = 0.60 * LG_RPG / LG_XWOBA  # peso xwOBA (0.60) x pendiente-ratio del TTE
    pa_per_game = pas.mean() / 162
    slope_linear_weights = pa_per_game / LG_WOBA_SCALE  # formula sabermetrica estandar, no usada hoy

    print(f"=== {season} (fin de temporada, encogimiento minimo) ===")
    print(f"  n equipos = {len(xw)}")
    print(f"  pendiente EMPIRICA (RS/G real ~ xwOBA):        {slope_emp:6.2f}  (corr={np.corrcoef(xw, rsg)[0, 1]:+.3f})")
    print(f"  pendiente del TTE actual (ratio x 0.60):        {slope_model:6.2f}")
    print(f"  pendiente 'linear weights' (PA/juego / wOBAscale, ya en el codigo, sin usar): {slope_linear_weights:6.2f}")
    print(f"  => el TTE actual es {slope_emp / slope_model:.2f}x MAS PLANO que la relacion real")
    print(f"  dispersion cruzada de equipos: RS/G real sd={rsg.std():.4f}  lambda TTE sd={comp.std():.4f}"
          f"  (ratio {comp.std() / rsg.std():.2f})")
    print(f"  corr(lambda TTE completo, RS/G real) = {np.corrcoef(comp, rsg)[0, 1]:+.3f}  (la DIRECCION es buena;"
          f" el problema es la MAGNITUD)\n")
