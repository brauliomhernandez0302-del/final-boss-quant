import { TierBadge } from "../shared/TierBadge";
import { DirectionalBar } from "../shared/DirectionalBar";
import type { Prediction } from "../../api/types";
import matchupStyles from "./matchup.module.css";
import styles from "./ValueBets.module.css";

interface Props {
  prediction: Prediction;
}

// Strips the emoji value_detector.py bakes into `tier` (e.g. "🔥 ULTRA
// VALUE") — TierBadge's color already carries that signal, so the emoji
// would just be visual noise duplicating it.
function stripEmoji(label: string): string {
  return label.replace(/^[^\w]+/, "").trim();
}

export function ValueBets({ prediction }: Props) {
  const bets = prediction.best_bets;
  const moneyline = prediction.metadata.value?.markets?.moneyline;

  return (
    <div>
      {moneyline && (
        <div className={styles.pinnacleRef}>
          Referencia Pinnacle: home {moneyline.pin_home?.toFixed(2) ?? "—"} / away{" "}
          {moneyline.pin_away?.toFixed(2) ?? "—"}
          {moneyline.fair_source !== "pinnacle" && (
            <span className={matchupStyles.emptyNote}> (sin línea Pinnacle para este juego)</span>
          )}
        </div>
      )}

      {bets.length === 0 ? (
        <div className={matchupStyles.emptyNote}>
          El pipeline no encontró oportunidades de valor positivo para este juego.
        </div>
      ) : (
        <div className={styles.table}>
          <div className={styles.headerRow}>
            <span>Mercado</span>
            <span>Tier</span>
            <span>EV</span>
            <span>Kelly</span>
            <span>Odds</span>
          </div>
          {bets.map((bet) => (
            <div key={`${bet.market}-${bet.side}-${bet.rank}`} className={styles.row}>
              <span className={styles.market}>{bet.side || bet.market}</span>
              <TierBadge grade={bet.tier_grade} label={stripEmoji(bet.tier)} />
              <DirectionalBar
                label=""
                value={bet.ev}
                center={0}
                domainHalf={10}
                formatValue={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`}
              />
              <span className="fbq-num">{(bet.kelly * 100).toFixed(2)}%</span>
              <span className="fbq-num">{bet.odds.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
