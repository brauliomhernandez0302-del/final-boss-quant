import { useEffect, useState } from "react";
import { useGamesList } from "../hooks/useGamesList";
import { useMatchup } from "../hooks/useMatchup";
import { MatchupDashboard } from "../components/matchup/MatchupDashboard";
import styles from "./MatchupPage.module.css";

export function MatchupPage() {
  const { games, loading: gamesLoading, error: gamesError } = useGamesList();
  const [selected, setSelected] = useState<number | null>(null);
  const { data, loading, error } = useMatchup(selected);

  useEffect(() => {
    if (selected === null && games.length > 0) {
      setSelected(games[0].game_pk);
    }
  }, [games, selected]);

  return (
    <div>
      <div className={styles.picker}>
        <label className={styles.pickerLabel} htmlFor="game-select">
          Juego
        </label>
        {gamesLoading && <span className={styles.status}>Cargando calendario…</span>}
        {gamesError && <span className={styles.statusError}>{gamesError}</span>}
        {!gamesLoading && !gamesError && (
          <select
            id="game-select"
            className={styles.select}
            value={selected ?? ""}
            onChange={(e) => setSelected(Number(e.target.value))}
          >
            {games.map((g) => (
              <option key={g.game_pk} value={g.game_pk}>
                {g.away_team} @ {g.home_team} — {g.status}
              </option>
            ))}
          </select>
        )}
      </div>

      {loading && <div className={styles.status}>Corriendo el pipeline (Monte Carlo + odds)…</div>}
      {error && <div className={styles.statusError}>Error: {error}</div>}
      {data && !data.error && <MatchupDashboard payload={data} />}
      {data?.error && <div className={styles.statusError}>{data.error}</div>}
    </div>
  );
}
