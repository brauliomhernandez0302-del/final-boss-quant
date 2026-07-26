import styles from "./DirectionalBar.module.css";

interface Props {
  label: string;
  value: number;
  /** The "no effect" reference point — 1.0 for multipliers, 0 for EV/edge %. */
  center?: number;
  /** Deviation from `center` that fills the full half-bar (clamped beyond). */
  domainHalf: number;
  /** Formatted text shown at the right, e.g. "1.028" or "+0.94%". */
  formatValue: (v: number) => string;
  /** Below this deviation, the bar renders as flat/neutral (no color claim). */
  flatEpsilon?: number;
}

// A single real reported number (a pipeline multiplier, an EV%, an edge%)
// rendered as a bar around a neutral center — direction and magnitude read
// at a glance, the exact value stays in mono next to it. Never fabricates a
// number: `value` is always something the caller already has from the API.
export function DirectionalBar({
  label,
  value,
  center = 1,
  domainHalf,
  formatValue,
  flatEpsilon = 0,
}: Props) {
  const deviation = value - center;
  const pct = Math.max(-1, Math.min(1, deviation / domainHalf));
  const isFlat = Math.abs(deviation) <= flatEpsilon;
  const isUp = !isFlat && deviation > 0;

  const fillStyle = isFlat
    ? { left: "50%", width: 0 }
    : pct >= 0
      ? { left: "50%", width: `${pct * 50}%` }
      : { left: `${50 + pct * 50}%`, width: `${-pct * 50}%` };

  return (
    <div className={`${styles.row} ${label ? "" : styles.rowNoLabel}`}>
      {label && <span className={styles.label}>{label}</span>}
      <div className={styles.track}>
        <div className={styles.axis} />
        <div
          className={`${styles.fill} ${isFlat ? styles.flat : isUp ? styles.up : styles.down}`}
          style={fillStyle}
        />
      </div>
      <span
        className={`fbq-num ${styles.value} ${
          isFlat ? styles.valueFlat : isUp ? styles.valueUp : styles.valueDown
        }`}
      >
        {formatValue(value)}
      </span>
    </div>
  );
}
