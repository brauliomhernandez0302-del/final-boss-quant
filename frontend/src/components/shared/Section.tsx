import type { ReactNode } from "react";
import styles from "./Section.module.css";

interface Props {
  title: string;
  right?: ReactNode;
  children: ReactNode;
  collapsible?: boolean;
  defaultOpen?: boolean;
  /** Stagger index for the entrance animation — each section fades+rises in
   * slightly after the previous one on first load. Purely cosmetic, capped
   * so a long page doesn't keep animating for seconds. */
  index?: number;
}

// The "◆ " prefix on most section titles is a real accent mark (colored,
// glowing), not plain text — split it out instead of leaving it inline and
// the same muted color as the label.
function TitleContent({ title }: { title: string }) {
  if (title.startsWith("◆ ")) {
    return (
      <>
        <span className={styles.marker}>◆</span> {title.slice(2)}
      </>
    );
  }
  return <>{title}</>;
}

// Every matchup-dashboard block is a separate <Section> — the owner asked
// for each block visually separated with its own divider and header,
// explicitly not folded into one continuous card.
export function Section({ title, right, children, collapsible, defaultOpen = true, index = 0 }: Props) {
  const delay = Math.min(index, 6) * 60;
  const style = { animationDelay: `${delay}ms` };

  if (!collapsible) {
    return (
      <section className={`${styles.section} fbq-animate-in`} style={style}>
        <div className={styles.header}>
          <h2 className={styles.title}><TitleContent title={title} /></h2>
          {right}
        </div>
        <div className={styles.body}>{children}</div>
      </section>
    );
  }

  return (
    <section className={`${styles.section} fbq-animate-in`} style={style}>
      <details open={defaultOpen}>
        <summary className={styles.summary}>
          <h2 className={styles.title}><TitleContent title={title} /></h2>
          {right}
        </summary>
        <div className={styles.body}>{children}</div>
      </details>
    </section>
  );
}
