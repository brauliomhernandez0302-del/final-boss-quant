import styles from "./DuelBar.module.css";

interface Props {
  label: string;
  awayValue: number;
  homeValue: number;
  formatValue: (v: number) => string;
  /** True when a lower number is the favorable one (e.g. a suppression multiplier). */
  lowerIsBetter?: boolean;
  domainHalf: number;
  center?: number;
}

// Face-off row: two real values (already in the API response) compared on a
// shared axis from a common center, winner marked by weight + a dot — not by
// a color-coded "good/bad" bar, which would collide with the waterfall's
// direction-of-runs convention used elsewhere on this page.
export function DuelBar({
  label,
  awayValue,
  homeValue,
  formatValue,
  lowerIsBetter = true,
  domainHalf,
  center = 1,
}: Props) {
  const awayDev = Math.abs(awayValue - center);
  const homeDev = Math.abs(homeValue - center);
  const awayPct = Math.max(0, Math.min(1, awayDev / domainHalf)) * 100;
  const homePct = Math.max(0, Math.min(1, homeDev / domainHalf)) * 100;

  const awayWins = lowerIsBetter ? awayValue < homeValue : awayValue > homeValue;
  const homeWins = lowerIsBetter ? homeValue < awayValue : homeValue > awayValue;

  return (
    <div className={styles.row}>
      <div className={`${styles.side} ${styles.sideAway}`}>
        <span className={`fbq-num ${styles.value} ${styles.valueAway} ${awayWins ? styles.winner : ""}`}>
          {awayWins && <span className={styles.winnerDot}>● </span>}
          {formatValue(awayValue)}
        </span>
        <div className={`${styles.track} ${styles.trackAway}`}>
          <div className={`${styles.fill} ${styles.fillAway}`} style={{ width: `${awayPct}%` }} />
        </div>
      </div>
      <span className={styles.label}>{label}</span>
      <div className={styles.side}>
        <div className={styles.track}>
          <div className={`${styles.fill} ${styles.fillHome}`} style={{ width: `${homePct}%` }} />
        </div>
        <span className={`fbq-num ${styles.value} ${homeWins ? styles.winner : ""}`}>
          {homeWins && <span className={styles.winnerDot}>● </span>}
          {formatValue(homeValue)}
        </span>
      </div>
    </div>
  );
}
