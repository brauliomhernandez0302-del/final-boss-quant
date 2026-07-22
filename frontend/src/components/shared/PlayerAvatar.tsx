import { useState } from "react";
import styles from "./PlayerAvatar.module.css";

interface Props {
  playerId: number | null | undefined;
  name: string;
  size?: number;
}

// mlbstatic.com headshot URL pattern — verified live during FASE 0 against
// real player_ids (701542 Will Warren, 696149 Bubba Chandler: both 200,
// image/jpeg). Falls back to an initial avatar, never a broken image —
// project rule: missing data is shown as missing, never a fake placeholder
// that reads as real.
export function PlayerAvatar({ playerId, name, size = 40 }: Props) {
  const [failed, setFailed] = useState(false);
  const initial = name.trim().charAt(0).toUpperCase() || "?";

  if (!playerId || failed) {
    return (
      <div
        className={styles.fallback}
        style={{ width: size, height: size, fontSize: size * 0.42 }}
        title={name}
      >
        {initial}
      </div>
    );
  }

  return (
    <img
      className={styles.avatar}
      style={{ width: size, height: size }}
      src={`https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto/v1/people/${playerId}/headshot/67/current`}
      alt={name}
      onError={() => setFailed(true)}
    />
  );
}
