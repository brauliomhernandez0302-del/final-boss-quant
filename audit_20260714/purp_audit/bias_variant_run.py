#!/usr/bin/env python3
"""Corre el backtest insignia con una variante del sesgo, SIN tocar producción.

Cada corrida usa su propia COPIA de predictions_history.db (--db-path) y su
propio directorio de reportes. El repo no se modifica: las variantes se aplican
parcheando el módulo en memoria, no editando el archivo congelado.

    python3 bias_variant_run.py <variante> <db_copia> <report_dir>

variantes:
  base       sin parche — ancla de identidad contra el canónico 0.24675/55.05%
  a_off      compute_multidim_bias ≡ 1.0. Es EXACTAMENTE lo que produciría
             filtrar source='live' en 2024/2025: esas temporadas tienen 0 filas
             live, así que la query quedaría bajo min_samples y el código
             retorna 1.0 por su propio camino.
  b_strict   clamp ±10% (era ±30%) y min_samples 30 (era 8 en multidim, 10 en
             team_bias), sin filtrar source.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/raulio")

variant, db_copy, report_dir = sys.argv[1], sys.argv[2], sys.argv[3]

import modules.baseball_module.calibration.learning_engine as le

if variant == "a_off":
    le.LearningEngine.compute_multidim_bias = lambda self, *a, **k: 1.0
    # compute_team_bias es el fallback final de compute_multidim_bias; con el
    # filtro live tampoco tendría filas, así que también cae a 1.0.
    le.LearningEngine.compute_team_bias = lambda self, *a, **k: 1.0

elif variant == "b_strict":
    le._BIAS_CLAMP = 0.10
    le._MIN_SAMPLES = 30
    _orig_multidim = le.LearningEngine.compute_multidim_bias

    def _multidim_strict(self, team, season, home_away="home", month=None,
                         min_samples=30, before_date=None, prediction_source="live"):
        return _orig_multidim(self, team, season, home_away, month,
                              min_samples, before_date, prediction_source)

    le.LearningEngine.compute_multidim_bias = _multidim_strict

elif variant != "base":
    raise SystemExit(f"variante desconocida: {variant}")

import backtest_and_retrain as bt

# Mismos flags que la corrida canónica (reporte_delta.md §Comando).
sys.argv = [
    "backtest_and_retrain.py",
    "--season", "2024,2025", "--use-full-pit",
    "--pitcher-pit-cache-db", "/home/raulio/data/pit_cache_pitcher.db",
    "--team-tte-pit-cache-db", "/home/raulio/data/pit_cache_merged.db",
    "--defense-pit-cache-db", "/home/raulio/data/pit_cache_merged.db",
    "--bullpen-pit-cache-db", "/home/raulio/data/pit_cache_merged.db",
    "--db-path", db_copy,
    "--report-dir", report_dir,
]
print(f"=== variante={variant}  db={db_copy} ===", flush=True)
bt.main()
