"""api/server.py — thin, read-only Flask JSON API for the React frontend.

Presentation layer only. Every route either:
  (a) calls modules/baseball_module/core/run_module.py exactly like
      track_record/publisher.py already does, with persist=False (never
      writes to game_outcomes — see CLAUDE.md's "live ledger is production"
      rule), or
  (b) reads public MLB Stats API schedule/roster data via
      api/mlb_presentation.py, fully independent of any prediction engine.

Out of scope, enforced by construction (nothing here imports these):
promote_calibration.py, backtest_and_retrain.py, learning_engine.py's write
paths, track_record/publisher.py's publish path, core/value_detector.py
(read-only consumer of its output only). No route ever calls
run_daily_picks.py or touches the quarantine ledger.

Run (dev):
    source mi_entorno/bin/activate
    FLASK_APP=api.server flask run --port 5000
"""
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request

from api.mlb_presentation import (
    fetch_bullpen_usage,
    fetch_pitcher_bio,
    find_scheduled_game,
    list_scheduled_games,
)
from config import PITCHER_ENGINE_WEIGHTS

app = Flask(__name__)

# Manual CORS for local dev (Vite on a different port) — no new dependency
# for a single-origin, read-only dev API. Tighten via FRONTEND_ORIGIN env var
# before this ever runs anywhere but localhost.
import os

_FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")


@app.after_request
def _add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = _FRONTEND_ORIGIN
    resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


def _json_safe(obj: Any) -> Any:
    """Recursively strip what run_module()'s raw metadata carries that
    json.dumps can't handle on its own: numpy scalars/arrays (MC sample
    arrays, stripped entirely — same reason track_record/publisher.py strips
    *_samples before its own json.dumps) and the raw ValueTier Enum
    (core/value_detector.py excludes it from best_bets' top-10 spread via its
    own 'tier_enum' key, but the underlying per-market dict under
    metadata.value.markets still carries it)."""
    if isinstance(obj, dict):
        return {
            k: _json_safe(v)
            for k, v in obj.items()
            if not (isinstance(k, str) and k.endswith("_samples"))
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "item") and callable(obj.item) and hasattr(obj, "dtype"):
        return obj.item()  # numpy scalar
    if hasattr(obj, "tolist") and callable(obj.tolist) and hasattr(obj, "dtype"):
        return obj.tolist()  # numpy array — defensive, *_samples already dropped above
    return obj


@app.route("/api/mlb/games")
def games():
    return jsonify(_json_safe(list_scheduled_games()))


@app.route("/api/mlb/matchup/<int:game_pk>")
def matchup(game_pk: int):
    from modules.baseball_module.core.run_module import run_module

    schedule = find_scheduled_game(game_pk)
    if schedule is None:
        return jsonify({"error": f"game_pk {game_pk} not in today/tomorrow's schedule"}), 404

    try:
        prediction = run_module(
            game_id=game_pk,
            use_hfa=True,
            use_pitcher=True,
            analyze_f5=True,
            persist=False,
        )
    except Exception as e:
        return jsonify({"error": f"pipeline error: {e}"}), 500

    home_pitcher_id = schedule.get("home_pitcher_id")
    away_pitcher_id = schedule.get("away_pitcher_id")

    payload = {
        "game_pk": game_pk,
        "schedule": schedule,
        "prediction": prediction,
        # Not "the bullpen roster" — the relievers the bullpen engine actually
        # weighted into total_mult, with one defined count. See
        # api/mlb_presentation.py::fetch_bullpen_usage (VAL-7.2).
        "bullpen_usage": {
            "home": fetch_bullpen_usage(schedule.get("home_team_id")),
            "away": fetch_bullpen_usage(schedule.get("away_team_id")),
        },
        "pitcher_bio": {
            "home": fetch_pitcher_bio(home_pitcher_id) if home_pitcher_id else {},
            "away": fetch_pitcher_bio(away_pitcher_id) if away_pitcher_id else {},
        },
        # The five sub-factor weights the pitcher engine actually combined with.
        # metadata.pitcher only carries the resulting multipliers, and
        # total_multiplier is a WEIGHTED SUM OF DELTAS, never a product
        # (audit_20260714/val_audit/reporte.md VAL-6):
        #     total = 1 + Σ wᵢ × (factorᵢ − 1)
        # Without the weights the UI cannot show per-factor contributions that
        # actually add up to that total, and would have to hardcode constants
        # that could silently drift from config.py. Read-only import of the same
        # dict the engine reads — nothing here touches the engine.
        "engine_weights": {"pitcher": dict(PITCHER_ENGINE_WEIGHTS)},
    }
    return jsonify(_json_safe(payload))


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)), debug=True)
