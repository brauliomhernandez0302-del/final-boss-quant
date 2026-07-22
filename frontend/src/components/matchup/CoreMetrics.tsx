import { ProbabilityBar } from "../shared/ProbabilityBar";
import type { Prediction, ScheduleDetail } from "../../api/types";
import matchupStyles from "./matchup.module.css";
import styles from "./CoreMetrics.module.css";

interface Props {
  prediction: Prediction;
  schedule: ScheduleDetail;
}

// Closes the 3-layer probability confusion diagnosed earlier this project
// (raw Monte Carlo -> Platt-1D "modelo", persisted at game_outcomes.p_home
// -> Platt-2D "decision", the one core/value_detector.py actually computed
// EV/Kelly from, only when a Pinnacle fair line was available). The BIG
// number is always the one that generated the actual pick; the other layer
// is labeled alongside it, never silently dropped. See CLAUDE.md's estado
// actual section and docs/PROTOCOLO_CLV_V1.md for the full history of this
// bug class.
export function CoreMetrics({ prediction, schedule }: Props) {
  const { probabilities, lambdas_history } = prediction;
  const moneyline = prediction.metadata.value?.markets?.moneyline;
  const finalLambda = lambdas_history.final;

  const modelHome = probabilities.p_home;
  const modelAway = probabilities.p_away;
  const hasDecisionLayer = Boolean(moneyline);
  const decisionHome = moneyline?.home.probability ?? modelHome;
  const decisionAway = moneyline?.away.probability ?? modelAway;
  const isMarketAdjusted = moneyline?.fair_source === "pinnacle";

  return (
    <div className={styles.wrap}>
      <div className={styles.probBlock}>
        <ProbabilityBar
          homePct={decisionHome}
          awayPct={decisionAway}
          homeLabel={schedule.home_team}
          awayLabel={schedule.away_team}
        />
        <div className={styles.layerNote}>
          {!hasDecisionLayer && (
            <span className={matchupStyles.emptyNote}>
              Sin mercado moneyline disponible — mostrando probabilidad del
              modelo (Platt-1D) únicamente.
            </span>
          )}
          {hasDecisionLayer && isMarketAdjusted && (
            <>
              <span className={styles.tag}>AJUSTADA MERCADO · Platt-2D</span>
              <span className={matchupStyles.emptyNote}>
                Modelo (Platt-1D): <span className="fbq-num">{(modelHome * 100).toFixed(1)}%</span> home
                / <span className="fbq-num">{(modelAway * 100).toFixed(1)}%</span> away
              </span>
            </>
          )}
          {hasDecisionLayer && !isMarketAdjusted && (
            <span className={styles.tag}>MODELO · Platt-1D (sin línea Pinnacle para ajustar)</span>
          )}
        </div>
      </div>

      <div className={matchupStyles.grid2}>
        <div className={matchupStyles.card}>
          <div className={matchupStyles.teamLabel}>λ esperadas</div>
          <div className={matchupStyles.statRow}>
            <span className={matchupStyles.statLabel}>{schedule.home_team}</span>
            <span className={`fbq-num ${matchupStyles.statValue}`}>{finalLambda.lh.toFixed(2)}</span>
          </div>
          <div className={matchupStyles.statRow}>
            <span className={matchupStyles.statLabel}>{schedule.away_team}</span>
            <span className={`fbq-num ${matchupStyles.statValue}`}>{finalLambda.la.toFixed(2)}</span>
          </div>
        </div>
        <div className={matchupStyles.card}>
          <div className={matchupStyles.teamLabel}>Total</div>
          <div className={matchupStyles.statRow}>
            <span className={matchupStyles.statLabel}>Proyectado</span>
            <span className={`fbq-num ${matchupStyles.statValue}`}>
              {probabilities.mean_total.toFixed(1)}
            </span>
          </div>
          {probabilities.total_line != null ? (
            <>
              <div className={matchupStyles.statRow}>
                <span className={matchupStyles.statLabel}>Línea de mercado</span>
                <span className={`fbq-num ${matchupStyles.statValue}`}>
                  {probabilities.total_line.toFixed(1)}
                </span>
              </div>
              <div className={matchupStyles.statRow}>
                <span className={matchupStyles.statLabel}>P(Over) / P(Under)</span>
                <span className={`fbq-num ${matchupStyles.statValue}`}>
                  {((probabilities.p_over ?? 0) * 100).toFixed(1)}% /{" "}
                  {((probabilities.p_under ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
            </>
          ) : (
            <span className={matchupStyles.emptyNote}>Sin línea de mercado disponible</span>
          )}
        </div>
      </div>
    </div>
  );
}
