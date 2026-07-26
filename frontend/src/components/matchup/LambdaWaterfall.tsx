import type { ReactNode } from "react";
import { DirectionalBar } from "../shared/DirectionalBar";
import type { Prediction, ScheduleDetail } from "../../api/types";
import matchupStyles from "./matchup.module.css";
import styles from "./LambdaWaterfall.module.css";

interface Props {
  prediction: Prediction;
  schedule: ScheduleDetail;
}

// Every step's domain is scaled on the SAME half-width so a tiny defense
// nudge (~1.001) and a large pitcher factor (~0.90) are honestly comparable
// by eye — using a different scale per row would flatter small effects.
const DOMAIN_HALF = 0.15;
const fmtMult = (v: number) => v.toFixed(3);

interface Step {
  key: string;
  label: string;
  /** The bar value — what the pipeline actually multiplied λ by. For
   * weighted stages this is 1 + w*(raw-1), NOT the raw engine ratio. */
  effective: number | null;
  /** Raw engine ratio, shown in the detail expansion (null = not weighted,
   * this step applies directly, e.g. Kalman/team-bias). */
  raw?: number | null;
  weight?: number | null;
  naReason?: string;
  detail?: ReactNode;
}

export function LambdaWaterfall({ prediction, schedule }: Props) {
  const { lambdas_history, metadata } = prediction;
  const { contextual, park_weather, hfa, pipeline_diagnostics: diag } = metadata;
  const finalLambda = lambdas_history.final;

  const talentAway = metadata.tte_away?.lambda_talent ?? null;
  const talentHome = metadata.tte_home?.lambda_talent ?? null;

  if (!diag) {
    return (
      <div className={matchupStyles.emptyNote}>
        Sin pipeline_diagnostics en esta respuesta — la cascada necesita un run_module() con la exposición aditiva
        (Kalman/sesgo/pesos) para mostrar el valor efectivo real.
      </div>
    );
  }

  const rr = diag.raw_ratios;
  const w = diag.pipeline_weights;
  const effOf = (raw: number, weight: number) => 1 + weight * (raw - 1);

  function weightedStep(key: string, label: string, rawKey: string, weightKey: keyof typeof w): Step {
    const raw = rr[rawKey];
    if (raw == null) return { key, label, effective: null, naReason: "sin dato" };
    const weight = w[weightKey];
    return { key, label, effective: effOf(raw, weight), raw, weight };
  }

  function makeSide(side: "away" | "home") {
    const pitcherKey = side === "away" ? "pitcher_on_away_lambda" : "pitcher_on_home_lambda";
    const contextKey = side === "away" ? "context_on_away_lambda" : "context_on_home_lambda";
    const bullpenKey = side === "away" ? "bullpen_on_away_lambda" : "bullpen_on_home_lambda";
    const parkKey = side === "away" ? "park_on_away_lambda" : "park_on_home_lambda";
    const defenseKey = side === "away" ? "defense_on_away_lambda" : "defense_on_home_lambda";
    const hfaKey = side === "away" ? "hfa_on_away_lambda" : "hfa_on_home_lambda";
    const restReason = side === "away" ? contextual?.away_rest_reason : contextual?.home_rest_reason;

    const steps: Step[] = [
      {
        key: "kalman",
        label: "Kalman offense",
        effective: diag.kalman_ratio[side],
      },
      {
        key: "bias",
        label: "Sesgo de equipo",
        effective: diag.team_bias[side],
      },
      weightedStep("pitcher", "Pitcher rival", pitcherKey, "pitcher"),
      { ...weightedStep("rest", `Descanso (${restReason ?? "?"})`, contextKey, "context") },
      weightedStep("bullpen", "Bullpen rival", bullpenKey, "bullpen"),
      {
        ...weightedStep("park_weather", "Parque & clima", parkKey, "park"),
        detail: park_weather && (
          <div className={styles.detailBody}>
            {park_weather.roof_closed ? (
              <span>Techo cerrado — clima no aplica.</span>
            ) : (
              <>
                <span>park_factor ×{park_weather.park_factor.toFixed(3)}</span>
                {park_weather.temp_f != null && (
                  <span>
                    {park_weather.temp_f.toFixed(1)}°F ×{park_weather.temp_mult.toFixed(4)}
                  </span>
                )}
                {park_weather.wind_mph != null && (
                  <span>
                    viento {park_weather.wind_mph.toFixed(1)}mph ×{park_weather.wind_mult.toFixed(4)}
                  </span>
                )}
                <span>lluvia ×{park_weather.rain_mult.toFixed(4)}</span>
              </>
            )}
          </div>
        ),
      },
      weightedStep("defense", "Defensa rival", defenseKey, "defense"),
      side === "home"
        ? weightedStep("hfa", "HFA (crowd, local)", hfaKey, "hfa")
        : rr[hfaKey] != null
          ? { ...weightedStep("hfa", "Fatiga de viaje (visitante)", hfaKey, "hfa") }
          : { key: "hfa", label: "HFA", effective: null, naReason: "sin dato de viaje" },
    ];
    return steps;
  }

  function Panel({
    teamName,
    roleTag,
    talent,
    steps,
    final,
  }: {
    teamName: string;
    roleTag: string;
    talent: number | null;
    steps: Step[];
    final: number;
  }) {
    return (
      <div className={styles.panel}>
        <div className={styles.panelHeader}>
          <span className={styles.team}>{teamName}</span>
          <span className={styles.roleTag}>{roleTag}</span>
        </div>

        <div className={styles.bookend}>
          <span className={styles.bookendLabel}>λ talento base</span>
          {talent != null ? (
            <span className={`fbq-num ${styles.bookendValue}`}>{talent.toFixed(3)}</span>
          ) : (
            <span className={matchupStyles.emptyNote}>sin dato de TTE</span>
          )}
        </div>
        <div className={styles.divider} />

        <div className={styles.steps}>
          {steps.map((s) =>
            s.effective != null ? (
              <div key={s.key}>
                <DirectionalBar
                  label={s.label}
                  value={s.effective}
                  domainHalf={DOMAIN_HALF}
                  formatValue={fmtMult}
                  flatEpsilon={0.0005}
                />
                {(s.raw != null || s.detail) && (
                  <details className={styles.detail}>
                    <summary className={styles.detailSummary}>crudo / peso ▾</summary>
                    <div className={styles.detailBody}>
                      {s.raw != null && (
                        <span>
                          raw ×{s.raw.toFixed(4)} · peso {s.weight!.toFixed(3)} → efectivo ×
                          {s.effective.toFixed(4)}
                        </span>
                      )}
                    </div>
                    {s.detail}
                  </details>
                )}
              </div>
            ) : (
              <div key={s.key} className={styles.na}>
                <span className={styles.naLabel}>{s.label}</span>
                <span />
                <span className={styles.naValue}>{s.naReason ?? "sin dato"}</span>
              </div>
            )
          )}
        </div>

        <div className={styles.divider} />
        <div className={styles.bookend}>
          <span className={styles.bookendLabel}>λ final</span>
          <span className={`fbq-num ${styles.bookendValue} ${styles.finalValue}`}>{final.toFixed(3)}</span>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className={styles.legend}>
        <span>
          <span className={styles.legendDot} style={{ background: "var(--fbq-accent-teal)" }} />
          sube λ (más carreras esperadas)
        </span>
        <span>
          <span className={styles.legendDot} style={{ background: "var(--fbq-accent-red)" }} />
          baja λ (menos carreras esperadas)
        </span>
        <span className={matchupStyles.emptyNote}>
          Barra = valor EFECTIVO aplicado (λ_out = λ_in × (1 + peso × (crudo − 1))) — el crudo y el peso del
          pipeline están en "crudo / peso" de cada paso. Kalman y sesgo se aplican directo (sin peso).
        </span>
      </div>
      <div className={styles.wrap}>
        <Panel
          teamName={schedule.away_team}
          roleTag="AWAY"
          talent={talentAway}
          steps={makeSide("away")}
          final={finalLambda.la}
        />
        <Panel
          teamName={schedule.home_team}
          roleTag="HOME"
          talent={talentHome}
          steps={makeSide("home")}
          final={finalLambda.lh}
        />
      </div>
    </div>
  );
}
