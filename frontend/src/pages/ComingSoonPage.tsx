interface Props {
  title: string;
}

// Deliberate placeholder — track record / CLV screens get designed AFTER
// there's real quarantine data to know what CLV metrics are actually
// useful (see project notes). This is scaffolding, not a guess at the
// eventual design.
export function ComingSoonPage({ title }: Props) {
  return (
    <div>
      <h1 style={{ fontFamily: "var(--fbq-font-sans)", fontSize: 20, marginBottom: 8 }}>{title}</h1>
      <p style={{ color: "var(--fbq-text-muted)", fontSize: 14 }}>
        Pantalla pendiente — se diseña sobre datos reales de cuarentena / CLV cuando existan.
      </p>
    </div>
  );
}
