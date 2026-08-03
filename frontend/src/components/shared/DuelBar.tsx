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
  /**
   * "deviation" (default): the bar measures |value − center| — for raw
   * multipliers, where direction is read from the number itself.
   *
   * "signed": `value` is already a signed delta around 0 (e.g. a weighted
   * contribution `w × (factor − 1)`). The bar measures |value| and the fill is
   * colored by direction using the SAME convention as the λ waterfall on this
   * page: teal = pushes the affected λ up, red = pushes it down.
   */
  mode?: "deviation" | "signed";
  /** In signed mode, |value| at or below this renders neutral (no color claim). */
  flatEpsilon?: number;
  /** Set false where marking a "winner" would be a judgment, not a fact. */
  markWinner?: boolean;
}

// Face-off row: two real values (already in the API response) compared on a
// shared axis from a common center. In deviation mode the winner is marked by
// weight + a dot rather than by a good/bad color, which would collide with the
// waterfall's direction-of-runs convention; in signed mode the color IS that
// same direction convention, so the two readings stay compatible.
export function DuelBar({
  label,
  awayValue,
  homeValue,
  formatValue,
  lowerIsBetter = true,
  domainHalf,
  center = 1,
  mode = "deviation",
  flatEpsilon = 0,
  markWinner = true,
}: Props) {
  const signed = mode === "signed";
  const origin = signed ? 0 : center;

  const awayDev = awayValue - origin;
  const homeDev = homeValue - origin;
  const awayPct = Math.max(0, Math.min(1, Math.abs(awayDev) / domainHalf)) * 100;
  const homePct = Math.max(0, Math.min(1, Math.abs(homeDev) / domainHalf)) * 100;

  const awayWins = markWinner && (lowerIsBetter ? awayValue < homeValue : awayValue > homeValue);
  const homeWins = markWinner && (lowerIsBetter ? homeValue < awayValue : homeValue > awayValue);

  const fillClass = (dev: number, sideClass: string) => {
    if (!signed) return sideClass;
    if (Math.abs(dev) <= flatEpsilon) return styles.fillFlat;
    return dev > 0 ? styles.fillUp : styles.fillDown;
  };

  return (
    <div className={styles.row}>
      <div className={`${styles.side} ${styles.sideAway}`}>
        <span className={`fbq-num ${styles.value} ${styles.valueAway} ${awayWins ? styles.winner : ""}`}>
          {awayWins && <span className={styles.winnerDot}>● </span>}
          {formatValue(awayValue)}
        </span>
        <div className={`${styles.track} ${styles.trackAway}`}>
          <div
            className={`${styles.fill} ${fillClass(awayDev, styles.fillAway)}`}
            style={{ width: `${awayPct}%` }}
          />
        </div>
      </div>
      <span className={styles.label}>{label}</span>
      <div className={styles.side}>
        <div className={styles.track}>
          <div
            className={`${styles.fill} ${fillClass(homeDev, styles.fillHome)}`}
            style={{ width: `${homePct}%` }}
          />
        </div>
        <span className={`fbq-num ${styles.value} ${homeWins ? styles.winner : ""}`}>
          {homeWins && <span className={styles.winnerDot}>● </span>}
          {formatValue(homeValue)}
        </span>
      </div>
    </div>
  );
}
