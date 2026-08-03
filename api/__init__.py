"""api/ — thin, read-only JSON layer for the React frontend.

Presentation only: composes data already produced elsewhere (run_module()'s
own pipeline output, MLB Stats API schedule/roster lookups) into one payload
per matchup. Never calls promote_calibration.py, never writes to
game_outcomes/ml_state, never touches track_record's publish path. See
api/server.py's module docstring for the full boundary.
"""
