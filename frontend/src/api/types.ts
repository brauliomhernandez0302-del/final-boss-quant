// Types mirror api/server.py's real payload shape (verified live against
// modules/baseball_module/core/run_module.py's actual output for game_pk
// 823519 during FASE 0 — not guessed from the pipeline's source alone).
// Fields not needed by the dashboard are left as `unknown`/optional rather
// than fully typed — the raw `metadata` blob backs the technical-details
// panel and is intentionally loose.

export interface ScheduleGame {
  game_pk: number;
  home_team: string;
  away_team: string;
  home_team_id: number;
  away_team_id: number;
  venue: string | null;
  status: string | null;
  game_date: string;
  official_date: string | null;
}

export interface LineupPlayer {
  id: number;
  name: string;
  position: string | null;
}

export interface ScheduleDetail extends ScheduleGame {
  home_pitcher: string | null;
  home_pitcher_id: number | null;
  away_pitcher: string | null;
  away_pitcher_id: number | null;
  home_lineup: LineupPlayer[];
  away_lineup: LineupPlayer[];
  // True only when MLB has posted both starting lineups (>=9 batters each
  // side) — see api/mlb_presentation.py::find_scheduled_game(). False means
  // "no lineup data at all yet", not "not confirmed but here's a projection".
  lineup_confirmed: boolean;
  is_playoff: boolean;
  game_type: string;
}

export interface Probabilities {
  n: number;
  p_home: number; // Platt-1D — "modelo" (also what game_outcomes.p_home persists)
  p_away: number;
  mean_home: number;
  mean_away: number;
  mean_total: number;
  std_total: number;
  total_line?: number;
  p_over?: number;
  p_under?: number;
  p_push?: number;
  p_rl_home?: number;
  p_rl_away?: number;
  f5_home?: number;
  f5_away?: number;
  f5_draw?: number;
  converged_early: boolean;
}

export interface LambdaSnapshot {
  lh: number;
  la: number;
}

export interface PitcherAdjustment {
  pitcher_name: string;
  quality_mult: number;
  form_mult: number;
  matchup_mult: number;
  platoon_mult: number;
  fatigue_mult: number;
  total_multiplier: number;
}

export interface BullpenAdjustment {
  era: number;
  effective_era: number;
  quality_mult: number;
  workload_mult: number;
  total_mult: number;
  n_pitchers: number;
  tier_label: string;
  used_siera: boolean;
  siera_ag: number | null;
}

export interface ValueBet {
  market: string;
  side: string;
  probability: number;
  odds: number;
  ev: number;
  ev_ci: [number, number];
  kelly: number;
  tier: string;
  tier_grade: string;
  edge: number;
  confidence: number;
  composite_score: number;
  line?: number;
  rank: number;
}

export interface MoneylineMarketSide {
  market: string;
  probability: number; // decision probability — Platt-2D when fair_source === "pinnacle"
  odds: number;
  ev: number;
  kelly: number;
  tier: string;
  tier_grade: string;
}

export interface MoneylineMarket {
  home: MoneylineMarketSide;
  away: MoneylineMarketSide;
  fair_source: string; // "pinnacle" when Platt-2D applied; otherwise a devig method name
  pin_home: number | null;
  pin_away: number | null;
  pin_fair_home: number | null;
  pin_fair_away: number | null;
}

export interface ValueMetadata {
  markets: {
    moneyline?: MoneylineMarket;
    total?: unknown;
    runline?: unknown;
    first5?: unknown;
  };
  global_recommendation: unknown;
  metadata: {
    fair_source: string;
    pin_home: number | null;
    pin_away: number | null;
  };
}

export interface PredictionMetadata {
  pitcher?: { pitcher_home: PitcherAdjustment; pitcher_away: PitcherAdjustment };
  bullpen?: { bullpen_home: BullpenAdjustment; bullpen_away: BullpenAdjustment };
  contextual?: Record<string, unknown>;
  defense?: Record<string, unknown>;
  hfa?: Record<string, unknown>;
  park_weather?: Record<string, unknown>;
  tte_home?: Record<string, unknown>;
  tte_away?: Record<string, unknown>;
  market_odds?: Record<string, unknown>;
  value?: ValueMetadata;
}

export interface Prediction {
  status: string;
  game_info: {
    home_team: string;
    away_team: string;
    pitcher_home: string;
    pitcher_away: string;
  };
  probabilities: Probabilities;
  lambdas_history: Record<string, LambdaSnapshot>;
  best_bets: ValueBet[];
  metadata: PredictionMetadata;
}

export interface RosterPitcher {
  id: number;
  name: string;
}

export interface MatchupPayload {
  game_pk: number;
  schedule: ScheduleDetail;
  prediction: Prediction;
  bullpen_roster: { home: RosterPitcher[]; away: RosterPitcher[] };
  pitcher_bio: { home?: { throws?: string }; away?: { throws?: string } };
  error?: string;
}
