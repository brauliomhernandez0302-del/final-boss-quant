import { useState } from "react";
import styles from "./TeamLogo.module.css";

interface Props {
  teamId: number | null | undefined;
  teamName: string;
  size?: number;
}

// mlbstatic.com team-logo URL pattern — verified live during FASE 0 (200,
// image/svg+xml for team_id 147/134). Falls back to team initials if the
// image ever 404s — never a broken-image icon.
export function TeamLogo({ teamId, teamName, size = 48 }: Props) {
  const [failed, setFailed] = useState(false);
  const initials = teamName
    .split(" ")
    .map((w) => w[0])
    .filter(Boolean)
    .slice(-2)
    .join("")
    .toUpperCase();

  if (!teamId || failed) {
    return (
      <div
        className={styles.fallback}
        style={{ width: size, height: size, fontSize: size * 0.36 }}
        title={teamName}
      >
        {initials}
      </div>
    );
  }

  return (
    <img
      className={styles.logo}
      src={`https://www.mlbstatic.com/team-logos/${teamId}.svg`}
      alt={`${teamName} logo`}
      width={size}
      height={size}
      onError={() => setFailed(true)}
    />
  );
}
