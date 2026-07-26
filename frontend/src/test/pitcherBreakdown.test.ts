import { describe, expect, it } from "vitest";
import { pitcherBreakdown } from "../components/matchup/StartingPitchers";
import type { PitcherAdjustment, PitcherEngineWeights } from "../api/types";

// config.py::PITCHER_ENGINE_WEIGHTS at the time of writing. The component reads
// these from the API, never from a constant — this copy exists only so the test
// can reproduce the audit's hand-computed cases.
const WEIGHTS: PitcherEngineWeights = {
  pitcher_quality: 0.321,
  pitcher_form: 0.256,
  pitcher_matchup: 0.192,
  pitcher_fatigue: 0.128,
  pitcher_platoon: 0.103,
};

function adjustment(over: Partial<PitcherAdjustment>): PitcherAdjustment {
  return {
    pitcher_name: "test",
    quality_mult: 1,
    form_mult: 1,
    matchup_mult: 1,
    platoon_mult: 1,
    fatigue_mult: 1,
    total_multiplier: 1,
    ...over,
  };
}

describe("pitcherBreakdown — VAL-6 additive decomposition", () => {
  // audit_20260714/val_audit/reporte.md VAL-6, Sugano (823759): the audit
  // reproduced total_multiplier = 1.164497683672599 by hand from these factors.
  it("reproduces the audited Sugano case", () => {
    const b = pitcherBreakdown(
      adjustment({
        quality_mult: 1.45,
        form_mult: 1.08519,
        matchup_mult: 1.0,
        platoon_mult: 0.97297,
        fatigue_mult: 1.008,
        total_multiplier: 1.164497683672599,
      }),
      WEIGHTS
    );
    expect(1 + b.sum).toBeCloseTo(1.1645, 5);
    expect(b.clamped).toBe(false);
  });

  // Same audit, Drohan: 0.9602661283907692 — the side that lands below 1.0.
  it("reproduces the audited Drohan case", () => {
    const b = pitcherBreakdown(
      adjustment({
        quality_mult: 0.84874,
        form_mult: 1.00773,
        matchup_mult: 1.0,
        platoon_mult: 1.05647,
        fatigue_mult: 1.008,
        total_multiplier: 0.9602661283907692,
      }),
      WEIGHTS
    );
    // 4 dp: the audit's published factors are rounded to 5 decimals, so the
    // last digit of the engine's own 0.9602661… can't be reproduced from them.
    expect(1 + b.sum).toBeCloseTo(0.9602661283907692, 4);
    expect(b.clamped).toBe(false);
  });

  it("the five contributions sum to the applied delta — the whole point of the bars", () => {
    const adj = adjustment({
      quality_mult: 1.12,
      form_mult: 0.94,
      matchup_mult: 1.03,
      platoon_mult: 0.98,
      fatigue_mult: 1.01,
    });
    const expected =
      1 +
      WEIGHTS.pitcher_quality * 0.12 +
      WEIGHTS.pitcher_form * -0.06 +
      WEIGHTS.pitcher_matchup * 0.03 +
      WEIGHTS.pitcher_platoon * -0.02 +
      WEIGHTS.pitcher_fatigue * 0.01;
    const b = pitcherBreakdown({ ...adj, total_multiplier: expected }, WEIGHTS);
    const addends = b.rows.reduce((acc, r) => acc + r.contribution, 0);
    expect(addends).toBeCloseTo(b.applied, 12);
    expect(b.clamped).toBe(false);
  });

  it("is a weighted SUM, not a product — the two differ and the sum is what ships", () => {
    const factors = { quality_mult: 1.45, form_mult: 1.2, matchup_mult: 0.9, platoon_mult: 1.1, fatigue_mult: 1.05 };
    const product = Object.values(factors).reduce((a, b) => a * b, 1);
    const b = pitcherBreakdown(adjustment({ ...factors, total_multiplier: 1 }), WEIGHTS);
    expect(1 + b.sum).toBeCloseTo(1.19315, 5);
    expect(product).toBeCloseTo(1.80873, 5);
    expect(Math.abs(product - (1 + b.sum))).toBeGreaterThan(0.5);
  });

  it("flags the engine's [0.65, 1.45] clamp instead of pretending the bars still add up", () => {
    const b = pitcherBreakdown(
      adjustment({ quality_mult: 3.0, form_mult: 2.5, total_multiplier: 1.45 }),
      WEIGHTS
    );
    expect(b.sum).toBeGreaterThan(0.45);
    expect(b.applied).toBeCloseTo(0.45, 12);
    expect(b.clamped).toBe(true);
  });
});
