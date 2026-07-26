"""Regression: the pipeline snapshot must survive numpy scalars.

`publish_mlb_picks()` serializes the raw `bet` dict from the value detector.
On 2026-07-24/25 that dict carried `kelly_floor_applied` as a `np.bool_`
(`kelly > kelly_pre_floor`, numpy float on the left), and `json.dumps` blew up
with "Object of type bool is not JSON serializable" — numpy 2.x names that
class `bool`, which made the message look self-contradictory. Three cron runs
died after doing all the pipeline work and published zero picks.

These tests pin the encoder, not the caller, so the guard holds no matter which
field starts arriving as a numpy type next.
"""
import json

import numpy as np
import pytest

from track_record.publisher import _json_default


def dumps(obj):
    return json.dumps(obj, default=_json_default)


def test_numpy_bool_is_serializable():
    # The exact shape of the live crash.
    flag = np.float64(0.01) > 0.0028
    assert type(flag).__module__ == "numpy"
    assert json.loads(dumps({"kelly_floor_applied": flag})) == {"kelly_floor_applied": True}


def test_numpy_scalars_round_trip_by_value():
    payload = {
        "kelly": np.float64(0.0125),
        "n": np.int64(50_000),
        "flag": np.bool_(False),
    }
    assert json.loads(dumps(payload)) == {"kelly": 0.0125, "n": 50000, "flag": False}


def test_numpy_array_becomes_a_list():
    assert json.loads(dumps({"pcts": np.array([1.0, 2.5])})) == {"pcts": [1.0, 2.5]}


def test_realistic_pipeline_snapshot_survives():
    snap = {
        "lambdas": {"final": {"lh": np.float64(4.766), "la": np.float64(3.183)}},
        "mc_probs": {"home_win": np.float64(0.646), "n": np.int64(1_000_000)},
        "bet": {
            "market": "RUNLINE HOME",
            "kelly_fraction": np.float64(0.01),
            "kelly_unfractional": 0.0112,
            "kelly_floor_applied": np.float64(0.01) > 0.0028,
            "prob_ci": [np.float64(0.61), np.float64(0.68)],
        },
    }
    restored = json.loads(dumps(snap))
    assert restored["bet"]["kelly_floor_applied"] is True
    assert restored["lambdas"]["final"]["lh"] == pytest.approx(4.766)


def test_genuinely_unserializable_still_raises():
    """The encoder must not turn a real bug into a silent string."""
    class Weird:
        pass

    with pytest.raises(TypeError, match="Weird"):
        dumps({"x": Weird()})
