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
  era_reg: number;
  effective_era: number;
  quality_mult: number;
  workload_mult: number;
  total_mult: number;
  // Three different "how many relievers" counts — none guaranteed equal,
  // see audit_20260714/val_audit/reporte.md VAL-7.2. n_pitchers = Savant
  // xwOBA/barrel coverage; n_siera_pitchers = FanGraphs SIERA/xFIP coverage;
  // n_reliever_ids = full classified roster before either coverage filter
  // (null when role classification itself failed, not just under-covered).
  n_pitchers: number;
  n_siera_pitchers?: number;
  n_reliever_ids?: number | null;
  tier_label: string;
  used_siera: boolean;
  siera_ag: number | null;
  xwoba_ag?: number;
  k_bb?: number;
  ip_3d?: number;
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

export interface ContextualAdjustment {
  home_rest_days: number;
  home_rest_mult: number;
  home_rest_reason: string;
  away_rest_days: number;
  away_rest_mult: number;
  away_rest_reason: string;
}

export interface TeamDefenseAdjustment {
  der_observed: number;
  der_regressed: number;
  der_factor: number;
  oaa: number;
  oaa_factor: number;
  bip_sample: number;
  games_played_est: number;
  final_mult: number;
  raw_mult: number;
}

export interface DefenseAdjustment {
  home_defense: TeamDefenseAdjustment;
  away_defense: TeamDefenseAdjustment;
  // Cross-mapped: the defense fields on this side apply to the OTHER team's
  // batting lambda (a team plays defense while its opponent bats).
  home_mult_on_away: number;
  away_mult_on_home: number;
}

export interface ParkWeatherAdjustment {
  park_name: string;
  park_factor: number;
  // Omitted entirely by the pipeline for closed-roof games (e.g. domes) —
  // weather doesn't apply, so these are NOT guaranteed present even though
  // temp_mult/wind_mult/rain_mult always are (they default to neutral 1.0).
  conditions?: string;
  temp_f?: number;
  temp_mult: number;
  wind_mph?: number;
  wind_dir?: number;
  wind_mult: number;
  rain_mult: number;
  weather_mult: number;
  total_mult: number;
  roof_closed: boolean;
  postponement_risk: boolean;
  weather_source?: string;
}

export interface HfaAdjustment {
  hfa_mult: number;
  hfa_boost_runs: number;
  uniform_home_mult: number;
  travel_penalty: number;
  travel_source?: string;
  park_name: string;
}

export interface TrueTalentAdjustment {
  team_id: number;
  team_name: string;
  season: number;
  lambda_talent: number;
  lambda_prior: number;
  lambda_cur: number;
  prior_weight: number;
  composite: number;
  factors: { f_xwoba: number; f_barrel: number; f_plate: number };
  metrics: {
    xwoba_raw: number;
    xwoba_regressed: number;
    barrel_pa_raw: number;
    barrel_regressed: number;
    bb_pct: number;
    k_pct: number;
    wrc_plus_approx: number;
  };
  lineup_confirmed: boolean;
  n_statcast_players: number;
  n_plate_discipline_players: number;
}

export interface PipelineDiagnostics {
  // The Kalman-offense step's own ratio (post/pre) — pulls TTE's talent λ
  // toward the team's observed run-scoring rate. Not a multiplier any
  // engine reports; run_module.py computes it locally and this is its
  // first exposure outside a log line.
  kalman_ratio: { home: number; away: number };
  // LearningEngine.compute_team_bias_kalman_adjusted() — applied right
  // after Kalman, before any of the 6 downstream engines, and never its
  // own lambdas_history stage (folded into what becomes that stage's
  // "_pre" value).
  team_bias: { home: number; away: number };
  // Learned per-stage weight w in λ_out = λ_in × (1 + w × (raw_ratio − 1)).
  // Keys match _STAGE_KEYS in learning_engine.py.
  pipeline_weights: {
    pitcher: number;
    context: number;
    bullpen: number;
    park: number;
    defense: number;
    hfa: number;
  };
  // run_module.py's own `_stage_factors` dict, exposed verbatim — the exact
  // raw ratio each stage computed (not re-derived/rounded for display), key
  // format "{stage}_on_{home,away}_lambda". Includes the away-side HFA key
  // (travel-fatigue ratio) even though there's no "hfa_away" multiplier
  // anywhere else in metadata — hfa.hfa_mult only ever covers the home
  // crowd-boost side.
  raw_ratios: Record<string, number>;
}

export interface PredictionMetadata {
  pitcher?: { pitcher_home: PitcherAdjustment; pitcher_away: PitcherAdjustment };
  bullpen?: { bullpen_home: BullpenAdjustment; bullpen_away: BullpenAdjustment };
  contextual?: ContextualAdjustment;
  defense?: DefenseAdjustment;
  hfa?: HfaAdjustment;
  park_weather?: ParkWeatherAdjustment;
  tte_home?: TrueTalentAdjustment;
  tte_away?: TrueTalentAdjustment;
  market_odds?: Record<string, unknown>;
  value?: ValueMetadata;
  pipeline_diagnostics?: PipelineDiagnostics;
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
