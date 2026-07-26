import styles from "./ProbabilityBar.module.css";

interface Props {
  homePct: number; // 0-1
  awayPct: number; // 0-1
  homeLabel: string;
  awayLabel: string;
}

// Hierarchy rule: the favored side's number is the hero — biggest, only one
// with the strong glow. The other side is present but visually secondary.
// Which side that is depends on the actual probabilities, not always "home".
export function ProbabilityBar({ homePct, awayPct, homeLabel, awayLabel }: Props) {
  const homeWidth = Math.round(homePct * 1000) / 10;
  const awayWidth = Math.round(awayPct * 1000) / 10;
  const homeFavored = homePct >= awayPct;

  return (
    <div className={styles.wrap}>
      <div className={styles.labels}>
        <span className={`${styles.side} ${homeFavored ? styles.hero : styles.secondary}`}>
          <span className={styles.name}>{homeLabel}</span>
          <span className={`fbq-num ${styles.pct}`}>{homeWidth.toFixed(1)}%</span>
        </span>
        <span className={`${styles.side} ${styles.sideRight} ${!homeFavored ? styles.hero : styles.secondary}`}>
          <span className={`fbq-num ${styles.pct}`}>{awayWidth.toFixed(1)}%</span>
          <span className={styles.name}>{awayLabel}</span>
        </span>
      </div>
      <div className={styles.track}>
        <div className={`${styles.fill} ${styles.home}`} style={{ width: `${homeWidth}%` }} />
        <div className={`${styles.fill} ${styles.away}`} style={{ width: `${awayWidth}%` }} />
      </div>
    </div>
  );
}
