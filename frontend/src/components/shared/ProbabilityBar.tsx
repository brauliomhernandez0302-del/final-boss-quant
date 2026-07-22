import styles from "./ProbabilityBar.module.css";

interface Props {
  homePct: number; // 0-1
  awayPct: number; // 0-1
  homeLabel: string;
  awayLabel: string;
}

export function ProbabilityBar({ homePct, awayPct, homeLabel, awayLabel }: Props) {
  const homeWidth = Math.round(homePct * 1000) / 10;
  const awayWidth = Math.round(awayPct * 1000) / 10;
  return (
    <div className={styles.wrap}>
      <div className={styles.labels}>
        <span className={styles.side}>
          <span className={styles.name}>{homeLabel}</span>
          <span className={`fbq-num ${styles.pct}`}>{homeWidth.toFixed(1)}%</span>
        </span>
        <span className={styles.side}>
          <span className={`fbq-num ${styles.pct}`}>{awayWidth.toFixed(1)}%</span>
          <span className={styles.name}>{awayLabel}</span>
        </span>
      </div>
      <div className={styles.track}>
        <div className={styles.home} style={{ width: `${homeWidth}%` }} />
        <div className={styles.away} style={{ width: `${awayWidth}%` }} />
      </div>
    </div>
  );
}
