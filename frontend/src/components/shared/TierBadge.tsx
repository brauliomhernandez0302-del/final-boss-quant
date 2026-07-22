import styles from "./TierBadge.module.css";

interface Props {
  grade: string; // "S" | "A" | "B" | "C" | ...
  label: string; // e.g. "ULTRA VALUE" — emoji stripped by the caller
}

const GRADE_CLASS: Record<string, string> = {
  S: styles.gradeS,
  A: styles.gradeA,
  B: styles.gradeB,
};

// Every grade renders at the SAME size/weight — color is the only signal.
// Per spec: an S-tier pick is a row, not a trophy. The backtest's own
// de-leaked baseline loses money in the high-edge bucket, so this
// deliberately does not add emphasis a human reader would read as "bet this."
export function TierBadge({ grade, label }: Props) {
  const cls = GRADE_CLASS[grade] ?? styles.gradeC;
  return (
    <span className={`${styles.badge} ${cls}`}>
      <span className={styles.grade}>{grade}</span>
      <span className={styles.label}>{label}</span>
    </span>
  );
}
