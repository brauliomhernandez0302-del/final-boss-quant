import styles from "./ProvenanceBadge.module.css";

interface Props {
  children: string;
}

// Project rule: missing/tentative data is signaled visibly, never filled
// with a placeholder that reads as real. This is the one visual vocabulary
// for that — UNCONFIRMED lineups, roster-not-availability-confirmed
// bullpen, missing weather/travel source, etc.
export function ProvenanceBadge({ children }: Props) {
  return <span className={styles.badge}>{children}</span>;
}
