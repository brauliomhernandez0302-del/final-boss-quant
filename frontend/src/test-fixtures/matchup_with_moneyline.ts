// SYNTHETIC fixture, for component tests only — never rendered in the real
// app. Built by taking the real captured payload (matchup_823519_fresh.json,
// live game_pk 823519, no moneyline market that call) and adding a
// moneyline block shaped exactly like core/value_detector.py's
// evaluate_value_ultra() actually returns it (metadata.value.markets.
// moneyline: {home, away, fair_source, pin_home, pin_away, ...}) — needed
// to exercise the Platt-2D "decision layer" labeling path, which the real
// capture didn't happen to have live odds for at the time.
import realPayload from "./matchup_823519_fresh.json";
import type { MatchupPayload } from "../api/types";

const base = realPayload as unknown as MatchupPayload;

export const matchupWithPinnacleMoneyline: MatchupPayload = {
  ...base,
  prediction: {
    ...base.prediction,
    best_bets: [
      {
        market: "MONEYLINE HOME",
        side: "HOME",
        probability: 0.612,
        odds: 1.81,
        ev: 6.4,
        ev_ci: [4.1, 8.7],
        kelly: 0.031,
        tier: "🟢 HIGH VALUE",
        tier_grade: "A",
        edge: 5.2,
        confidence: 0.9,
        composite_score: 45.2,
        rank: 1,
      },
    ],
    metadata: {
      ...base.prediction.metadata,
      value: {
        markets: {
          moneyline: {
            home: {
              market: "MONEYLINE HOME",
              probability: 0.612, // Platt-2D decision probability
              odds: 1.81,
              ev: 6.4,
              kelly: 0.031,
              tier: "🟢 HIGH VALUE",
              tier_grade: "A",
            },
            away: {
              market: "MONEYLINE AWAY",
              probability: 0.388,
              odds: 2.22,
              ev: -1.2,
              kelly: 0,
              tier: "⚪ NEUTRAL",
              tier_grade: "C",
            },
            fair_source: "pinnacle",
            pin_home: 1.8,
            pin_away: 2.16,
            pin_fair_home: 0.596,
            pin_fair_away: 0.404,
          },
        },
        global_recommendation: {},
        metadata: { fair_source: "pinnacle", pin_home: 1.8, pin_away: 2.16 },
      },
    },
  },
};

export const matchupNoMoneyline: MatchupPayload = base;
