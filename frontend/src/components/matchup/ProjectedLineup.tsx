import { PlayerAvatar } from "../shared/PlayerAvatar";
import { ProvenanceBadge } from "../shared/ProvenanceBadge";
import type { LineupPlayer, ScheduleDetail } from "../../api/types";
import matchupStyles from "./matchup.module.css";

interface Props {
  schedule: ScheduleDetail;
}

function LineupList({ team, lineup }: { team: string; lineup: LineupPlayer[] }) {
  return (
    <div className={matchupStyles.card}>
      <div className={matchupStyles.teamLabel}>{team}</div>
      {lineup.length === 0 ? (
        <div className={matchupStyles.emptyNote}>
          MLB todavía no ha publicado el lineup para este juego.
        </div>
      ) : (
        <div className={matchupStyles.list}>
          {lineup.map((p, i) => (
            <div key={p.id} className={matchupStyles.playerRow}>
              <span className={matchupStyles.playerMeta}>{i + 1}</span>
              <PlayerAvatar playerId={p.id} name={p.name} size={32} />
              <span className={matchupStyles.playerName}>{p.name}</span>
              {p.position && <span className={matchupStyles.playerMeta}>{p.position}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ProjectedLineup({ schedule }: Props) {
  return (
    <div>
      <div style={{ marginBottom: 12 }}>
        {schedule.lineup_confirmed ? (
          <span style={{ color: "var(--fbq-accent-teal)", fontSize: 12, fontWeight: 600 }}>
            ✓ LINEUP CONFIRMADO POR MLB
          </span>
        ) : (
          <ProvenanceBadge>UNCONFIRMED</ProvenanceBadge>
        )}
      </div>
      <div className={matchupStyles.grid2}>
        <LineupList team={schedule.away_team} lineup={schedule.away_lineup} />
        <LineupList team={schedule.home_team} lineup={schedule.home_lineup} />
      </div>
    </div>
  );
}
