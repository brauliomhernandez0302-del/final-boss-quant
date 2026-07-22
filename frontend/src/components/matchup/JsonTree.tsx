import styles from "./JsonTree.module.css";

// Minimal recursive viewer for the raw per-engine metadata — deliberately
// generic rather than hand-mapping every engine's fields, since this panel
// exists for advanced users who want the actual pipeline output, not a
// second curated summary.
export function JsonTree({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value === null || value === undefined) {
    return <span className={styles.null}>null</span>;
  }
  if (typeof value === "number") {
    return <span className={`fbq-num ${styles.number}`}>{value}</span>;
  }
  if (typeof value === "boolean") {
    return <span className={styles.bool}>{String(value)}</span>;
  }
  if (typeof value === "string") {
    return <span className={styles.string}>"{value}"</span>;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className={styles.null}>[]</span>;
    return (
      <div className={styles.indent}>
        {value.map((v, i) => (
          <div key={i} className={styles.line}>
            <JsonTree value={v} depth={depth + 1} />
          </div>
        ))}
      </div>
    );
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) return <span className={styles.null}>{"{}"}</span>;
    return (
      <div className={styles.indent}>
        {entries.map(([k, v]) => (
          <div key={k} className={styles.line}>
            <span className={styles.key}>{k}</span>
            <span className={styles.colon}>: </span>
            <JsonTree value={v} depth={depth + 1} />
          </div>
        ))}
      </div>
    );
  }
  return <span>{String(value)}</span>;
}
