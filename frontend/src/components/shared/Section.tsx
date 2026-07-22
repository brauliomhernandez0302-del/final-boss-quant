import type { ReactNode } from "react";
import styles from "./Section.module.css";

interface Props {
  title: string;
  right?: ReactNode;
  children: ReactNode;
  collapsible?: boolean;
  defaultOpen?: boolean;
}

// Every matchup-dashboard block is a separate <Section> — the owner asked
// for each block visually separated with its own divider and header,
// explicitly not folded into one continuous card.
export function Section({ title, right, children, collapsible, defaultOpen = true }: Props) {
  if (!collapsible) {
    return (
      <section className={styles.section}>
        <div className={styles.header}>
          <h2 className={styles.title}>{title}</h2>
          {right}
        </div>
        <div className={styles.body}>{children}</div>
      </section>
    );
  }

  return (
    <section className={styles.section}>
      <details open={defaultOpen}>
        <summary className={styles.summary}>
          <h2 className={styles.title}>{title}</h2>
          {right}
        </summary>
        <div className={styles.body}>{children}</div>
      </details>
    </section>
  );
}
