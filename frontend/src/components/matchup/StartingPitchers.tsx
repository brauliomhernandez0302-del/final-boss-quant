import { PlayerAvatar } from "../shared/PlayerAvatar";
import { DuelBar } from "../shared/DuelBar";
import type { MatchupPayload, PitcherAdjustment, PitcherEngineWeights } from "../../api/types";
import matchupStyles from "./matchup.module.css";
import styles from "./StartingPitchers.module.css";

interface Props {
  payload: MatchupPayload;
}

// The five sub-factors, in the engine's own weight order (config.py).
const FACTORS = [
  { key: "quality_mult", weightKey: "pitcher_quality", label: "Calidad" },
  { key: "form_mult", weightKey: "pitcher_form", label: "Forma reciente" },
  { key: "matchup_mult", weightKey: "pitcher_matchup", label: "Historial vs rival" },
  { key: "fatigue_mult", weightKey: "pitcher_fatigue", label: "Fatiga" },
  { key: "platoon_mult", weightKey: "pitcher_platoon", label: "Platoon" },
] as const satisfies readonly {
  key: keyof PitcherAdjustment;
  weightKey: keyof PitcherEngineWeights;
  label: string;
}[];

// pitcher_engine.py::_calculate_pitcher_adjustment clamps the combined result.
const CLAMP_LO = 0.65;
const CLAMP_HI = 1.45;

// Below this the sum and the applied total are the same number to the precision
// shown; above it the clamp really did bite and the UI has to say so.
const CLAMP_EPS = 5e-5;

const MULT_FMT = (v: number) => v.toFixed(3);
const DELTA_FMT = (v: number) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(4)}`;

export interface PitcherContribution {
  key: keyof PitcherAdjustment;
  label: string;
  weight: number;
  raw: number;
  /** wᵢ × (rawᵢ − 1) — the addend, in λ-multiplier units. */
  contribution: number;
}

export interface PitcherBreakdown {
  rows: PitcherContribution[];
  /** Σ of the five contributions — the delta BEFORE the engine's clamp. */
  sum: number;
  /** total_multiplier − 1 — the delta the engine actually applied. */
  applied: number;
  /** True when the clamp to [0.65, 1.45] moved the applied delta off the sum. */
  clamped: boolean;
}

/**
 * Decomposes one pitcher's `total_multiplier` into the five addends the engine
 * summed to build it (VAL-6): `total = 1 + Σ wᵢ × (factorᵢ − 1)`. Every number
 * here comes from the API (factors) or from the engine's own weight table
 * (echoed by the API) — nothing is fitted, inferred, or invented client-side.
 */
export function pitcherBreakdown(adj: PitcherAdjustment, weights: PitcherEngineWeights): PitcherBreakdown {
  const rows = FACTORS.map((f) => {
    const weight = weights[f.weightKey];
    const raw = adj[f.key] as number;
    return { key: f.key, label: f.label, weight, raw, contribution: weight * (raw - 1) };
  });
  const sum = rows.reduce((acc, r) => acc + r.contribution, 0);
  const applied = adj.total_multiplier - 1;
  return { rows, sum, applied, clamped: Math.abs(sum - applied) > CLAMP_EPS };
}

function sumExpression(contributions: number[]): string {
  return contributions
    .map((c, i) => {
      const mag = Math.abs(c).toFixed(4);
      if (i === 0) return c < 0 ? `−${mag}` : mag;
      return c < 0 ? ` − ${mag}` : ` + ${mag}`;
    })
    .join("");
}

export function StartingPitchers({ payload }: Props) {
  const { schedule, prediction, pitcher_bio, engine_weights } = payload;
  const pitcherMeta = prediction.metadata.pitcher;
  const away = pitcherMeta?.pitcher_away;
  const home = pitcherMeta?.pitcher_home;
  const weights = engine_weights?.pitcher;

  const header = (
    <div className={styles.header}>
      <div className={styles.playerCol}>
        <PlayerAvatar playerId={schedule.away_pitcher_id} name={schedule.away_pitcher ?? "TBD"} size={52} />
        <div>
          <div className={matchupStyles.playerName}>{schedule.away_pitcher ?? "TBD"}</div>
          <div className={matchupStyles.playerMeta}>
            {schedule.away_team}
            {pitcher_bio.away?.throws ? ` · ${pitcher_bio.away.throws}HP` : ""}
          </div>
          <div className={styles.roleTag}>ajusta λ de {schedule.home_team}</div>
        </div>
      </div>
      <span className={styles.vs}>VS</span>
      <div className={`${styles.playerCol} ${styles.playerColRight}`}>
        <div className={styles.textRight}>
          <div className={matchupStyles.playerName}>{schedule.home_pitcher ?? "TBD"}</div>
          <div className={matchupStyles.playerMeta}>
            {schedule.home_team}
            {pitcher_bio.home?.throws ? ` · ${pitcher_bio.home.throws}HP` : ""}
          </div>
          <div className={styles.roleTag}>ajusta λ de {schedule.away_team}</div>
        </div>
        <PlayerAvatar playerId={schedule.home_pitcher_id} name={schedule.home_pitcher ?? "TBD"} size={52} />
      </div>
    </div>
  );

  if (!away || !home) {
    return (
      <div className={styles.wrap}>
        {header}
        <div className={matchupStyles.emptyNote} style={{ marginTop: 12 }}>
          Sin ajuste de Pitcher Engine disponible para este juego.
        </div>
      </div>
    );
  }

  // Without the real weights, showing contributions would mean inventing them.
  // Fall back to the raw multipliers and say exactly what is missing.
  if (!weights) {
    return (
      <div className={styles.wrap}>
        {header}
        <div className={styles.rows}>
          {FACTORS.map((f) => (
            <DuelBar
              key={f.key}
              label={f.key}
              awayValue={away[f.key] as number}
              homeValue={home[f.key] as number}
              formatValue={MULT_FMT}
              lowerIsBetter
              domainHalf={0.2}
            />
          ))}
          <div className={styles.totalDivider} />
          <div className={styles.emphasizedRow}>
            <DuelBar
              label="total_multiplier"
              awayValue={away.total_multiplier}
              homeValue={home.total_multiplier}
              formatValue={MULT_FMT}
              lowerIsBetter
              domainHalf={0.2}
            />
          </div>
        </div>
        <div className={matchupStyles.emptyNote} style={{ marginTop: 8 }}>
          Esta respuesta del API no trae <code>engine_weights.pitcher</code>, así que se muestran los
          multiplicadores crudos: sin los pesos reales no se pueden mostrar aportes que sumen al total sin
          inventarlos.
        </div>
      </div>
    );
  }

  // VAL-6: total = 1 + Σ wᵢ × (factorᵢ − 1). Never a product — the bars below
  // are the addends of that exact sum, on one shared scale, so what the eye
  // adds up is what the engine added up.
  const awayBreak = pitcherBreakdown(away, weights);
  const homeBreak = pitcherBreakdown(home, weights);
  const rows = awayBreak.rows.map((r, i) => ({
    ...r,
    awayRaw: r.raw,
    homeRaw: homeBreak.rows[i].raw,
    awayContribution: r.contribution,
    homeContribution: homeBreak.rows[i].contribution,
  }));

  const { sum: awaySum, applied: awayApplied, clamped: awayClamped } = awayBreak;
  const { sum: homeSum, applied: homeApplied, clamped: homeClamped } = homeBreak;
  const weightSum = FACTORS.reduce((acc, f) => acc + weights[f.weightKey], 0);

  // One scale for every bar including the total, so the total bar is visibly
  // the sum of the others instead of being re-normalized to look bigger.
  const domainHalf = Math.max(
    0.02,
    ...rows.flatMap((r) => [Math.abs(r.awayContribution), Math.abs(r.homeContribution)]),
    Math.abs(awayApplied),
    Math.abs(homeApplied)
  );

  return (
    <div className={styles.wrap}>
      {header}

      <div className={styles.legend}>
        <span>
          <span className={styles.legendDot} style={{ background: "var(--fbq-accent-teal)" }} />
          sube la λ del rival
        </span>
        <span>
          <span className={styles.legendDot} style={{ background: "var(--fbq-accent-red)" }} />
          la baja
        </span>
        <span className={matchupStyles.emptyNote}>
          Barra = aporte = peso × (crudo − 1). Los cinco aportes SUMAN el delta total:{" "}
          <code>total = 1 + Σ peso × (crudo − 1)</code> — combinación lineal, nunca un producto. El crudo y el
          peso de cada factor están en su detalle.
        </span>
      </div>

      <div className={styles.rows}>
        {rows.map((r) => (
          <div key={r.key}>
            <DuelBar
              label={`${r.label} · w ${r.weight.toFixed(3)}`}
              awayValue={r.awayContribution}
              homeValue={r.homeContribution}
              formatValue={DELTA_FMT}
              mode="signed"
              domainHalf={domainHalf}
              flatEpsilon={5e-5}
              lowerIsBetter
            />
            <details className={styles.detail}>
              <summary className={styles.detailSummary}>crudo / peso ▾</summary>
              <div className={styles.detailBody}>
                <span>
                  {schedule.away_pitcher ?? "AWAY"}: crudo ×{r.awayRaw.toFixed(4)} · peso {r.weight.toFixed(3)}{" "}
                  → aporte {DELTA_FMT(r.awayContribution)}
                </span>
                <span>
                  {schedule.home_pitcher ?? "HOME"}: crudo ×{r.homeRaw.toFixed(4)} · peso {r.weight.toFixed(3)}{" "}
                  → aporte {DELTA_FMT(r.homeContribution)}
                </span>
              </div>
            </details>
          </div>
        ))}

        <div className={styles.totalDivider} />

        <div className={styles.emphasizedRow}>
          <DuelBar
            label="Δ total aplicado"
            awayValue={awayApplied}
            homeValue={homeApplied}
            formatValue={DELTA_FMT}
            mode="signed"
            domainHalf={domainHalf}
            flatEpsilon={5e-5}
            lowerIsBetter
          />
        </div>
      </div>

      <div className={styles.identity}>
        <div className={styles.identityRow}>
          <span className={styles.identitySide}>{schedule.away_team}</span>
          <span className={`fbq-num ${styles.identityMath}`}>
            {sumExpression(rows.map((r) => r.awayContribution))} = {DELTA_FMT(awaySum)} → ×
            {(1 + awaySum).toFixed(4)}
            {awayClamped && (
              <span className={styles.clamp}>
                {" "}
                · recortado a [{CLAMP_LO}, {CLAMP_HI}] → ×{away.total_multiplier.toFixed(4)} aplicado
              </span>
            )}
          </span>
        </div>
        <div className={styles.identityRow}>
          <span className={styles.identitySide}>{schedule.home_team}</span>
          <span className={`fbq-num ${styles.identityMath}`}>
            {sumExpression(rows.map((r) => r.homeContribution))} = {DELTA_FMT(homeSum)} → ×
            {(1 + homeSum).toFixed(4)}
            {homeClamped && (
              <span className={styles.clamp}>
                {" "}
                · recortado a [{CLAMP_LO}, {CLAMP_HI}] → ×{home.total_multiplier.toFixed(4)} aplicado
              </span>
            )}
          </span>
        </div>
      </div>

      <div className={matchupStyles.emptyNote} style={{ marginTop: 8 }}>
        Aporte negativo = ese factor suprime carreras del rival; no es un juicio de "bueno/malo" fuera de ese
        eje. Los cinco pesos suman {weightSum.toFixed(3)} y vienen del engine
        (<code>config.py::PITCHER_ENGINE_WEIGHTS</code>), no del frontend. Este delta total entra después al
        pipeline con su propio peso — esa segunda capa está en la cascada de λ.
      </div>
    </div>
  );
}
