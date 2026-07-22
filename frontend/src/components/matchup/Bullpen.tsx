import { PlayerAvatar } from "../shared/PlayerAvatar";
import { ProvenanceBadge } from "../shared/ProvenanceBadge";
import type { BullpenAdjustment, MatchupPayload, RosterPitcher } from "../../api/types";
import matchupStyles from "./matchup.module.css";

interface Props {
  payload: MatchupPayload;
}

function BullpenCard({
  team,
  roster,
  meta,
}: {
  team: string;
  roster: RosterPitcher[];
  meta: BullpenAdjustment | undefined;
}) {
  return (
    <div className={matchupStyles.card}>
      <div className={matchupStyles.teamLabel}>{team}</div>

      {meta ? (
        <div style={{ marginBottom: 12 }}>
          <div className={matchupStyles.statRow}>
            <span className={matchupStyles.statLabel}>
              {meta.used_siera ? "SIERA" : "ERA"}
            </span>
            <span className={`fbq-num ${matchupStyles.statValue}`}>
              {(meta.used_siera ? meta.siera_ag : meta.era)?.toFixed(2)}
            </span>
          </div>
          <div className={matchupStyles.statRow}>
            <span className={matchupStyles.statLabel}>ERA efectiva</span>
            <span className={`fbq-num ${matchupStyles.statValue}`}>{meta.effective_era.toFixed(2)}</span>
          </div>
          <div className={matchupStyles.statRow}>
            <span className={matchupStyles.statLabel}>total_mult</span>
            <span className={`fbq-num ${matchupStyles.statValue}`}>{meta.total_mult.toFixed(3)}</span>
          </div>
          <div className={matchupStyles.statRow}>
            <span className={matchupStyles.statLabel}>n disponibles (pipeline)</span>
            <span className={`fbq-num ${matchupStyles.statValue}`}>{meta.n_pitchers}</span>
          </div>
        </div>
      ) : (
        <div className={matchupStyles.emptyNote} style={{ marginBottom: 12 }}>
          Sin ajuste de Bullpen Engine disponible para este equipo.
        </div>
      )}

      <div className={matchupStyles.teamLabel} style={{ marginTop: 4 }}>
        Roster activo <ProvenanceBadge>ROSTER — NO CONFIRMADO</ProvenanceBadge>
      </div>
      {roster.length === 0 ? (
        <div className={matchupStyles.emptyNote}>Sin datos de roster disponibles.</div>
      ) : (
        <div className={matchupStyles.list}>
          {roster.map((p) => (
            <div key={p.id} className={matchupStyles.playerRow}>
              <PlayerAvatar playerId={p.id} name={p.name} size={32} />
              <span className={matchupStyles.playerName}>{p.name}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function Bullpen({ payload }: Props) {
  const { schedule, prediction, bullpen_roster } = payload;
  const bullpenMeta = prediction.metadata.bullpen;

  return (
    <div className={matchupStyles.grid2}>
      <BullpenCard
        team={schedule.away_team}
        roster={bullpen_roster.away}
        meta={bullpenMeta?.bullpen_away}
      />
      <BullpenCard
        team={schedule.home_team}
        roster={bullpen_roster.home}
        meta={bullpenMeta?.bullpen_home}
      />
    </div>
  );
}
