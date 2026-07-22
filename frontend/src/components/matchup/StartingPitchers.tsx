import { PlayerAvatar } from "../shared/PlayerAvatar";
import type { MatchupPayload } from "../../api/types";
import matchupStyles from "./matchup.module.css";

interface Props {
  payload: MatchupPayload;
}

function PitcherCard({
  name,
  playerId,
  team,
  throwsHand,
  mult,
}: {
  name: string;
  playerId: number | null | undefined;
  team: string;
  throwsHand: string | undefined;
  mult: { quality_mult: number; form_mult: number; total_multiplier: number } | undefined;
}) {
  return (
    <div className={matchupStyles.card}>
      <div className={matchupStyles.playerRow}>
        <PlayerAvatar playerId={playerId} name={name} size={56} />
        <div>
          <div className={matchupStyles.playerName}>{name}</div>
          <div className={matchupStyles.playerMeta}>
            {team}
            {throwsHand ? ` · ${throwsHand}HP` : ""}
          </div>
        </div>
      </div>
      {mult ? (
        <div style={{ marginTop: 12 }}>
          <div className={matchupStyles.statRow}>
            <span className={matchupStyles.statLabel}>quality_mult</span>
            <span className={`fbq-num ${matchupStyles.statValue}`}>{mult.quality_mult.toFixed(3)}</span>
          </div>
          <div className={matchupStyles.statRow}>
            <span className={matchupStyles.statLabel}>form_mult</span>
            <span className={`fbq-num ${matchupStyles.statValue}`}>{mult.form_mult.toFixed(3)}</span>
          </div>
          <div className={matchupStyles.statRow}>
            <span className={matchupStyles.statLabel}>total_multiplier</span>
            <span className={`fbq-num ${matchupStyles.statValue}`}>{mult.total_multiplier.toFixed(3)}</span>
          </div>
        </div>
      ) : (
        <div className={matchupStyles.emptyNote} style={{ marginTop: 12 }}>
          Sin ajuste de Pitcher Engine disponible para este juego.
        </div>
      )}
    </div>
  );
}

export function StartingPitchers({ payload }: Props) {
  const { schedule, prediction, pitcher_bio } = payload;
  const pitcherMeta = prediction.metadata.pitcher;

  return (
    <div className={matchupStyles.grid2}>
      <PitcherCard
        name={schedule.away_pitcher ?? "TBD"}
        playerId={schedule.away_pitcher_id}
        team={schedule.away_team}
        throwsHand={pitcher_bio.away?.throws}
        mult={pitcherMeta?.pitcher_away}
      />
      <PitcherCard
        name={schedule.home_pitcher ?? "TBD"}
        playerId={schedule.home_pitcher_id}
        team={schedule.home_team}
        throwsHand={pitcher_bio.home?.throws}
        mult={pitcherMeta?.pitcher_home}
      />
    </div>
  );
}
