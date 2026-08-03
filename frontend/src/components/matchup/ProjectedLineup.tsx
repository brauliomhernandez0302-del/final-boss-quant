import { PlayerAvatar } from "../shared/PlayerAvatar";
import { ProvenanceBadge } from "../shared/ProvenanceBadge";
import { DirectionalBar } from "../shared/DirectionalBar";
import type { LineupPlayer, Prediction, ScheduleDetail, TrueTalentAdjustment } from "../../api/types";
import matchupStyles from "./matchup.module.css";
import styles from "./ProjectedLineup.module.css";

interface Props {
  schedule: ScheduleDetail;
  prediction: Prediction;
}

function TeamOffenseAggregate({ tte }: { tte: TrueTalentAdjustment | undefined }) {
  if (!tte) {
    return <div className={matchupStyles.emptyNote}>Sin agregado de True Talent Engine.</div>;
  }
  return (
    <div className={styles.aggregate}>
      <DirectionalBar
        label="f_xwoba"
        value={tte.factors.f_xwoba}
        domainHalf={0.15}
        formatValue={(v) => v.toFixed(3)}
      />
      <DirectionalBar
        label="f_barrel"
        value={tte.factors.f_barrel}
        domainHalf={0.25}
        formatValue={(v) => v.toFixed(3)}
      />
      <DirectionalBar
        label="f_plate"
        value={tte.factors.f_plate}
        domainHalf={0.15}
        formatValue={(v) => v.toFixed(3)}
      />
      <div className={matchupStyles.statRow}>
        <span className={matchupStyles.statLabel}>wRC+ aprox.</span>
        <span className={`fbq-num ${matchupStyles.statValue}`}>{tte.metrics.wrc_plus_approx.toFixed(1)}</span>
      </div>
      <div className={matchupStyles.statRow}>
        <span className={matchupStyles.statLabel}>xwOBA regresado</span>
        <span className={`fbq-num ${matchupStyles.statValue}`}>{tte.metrics.xwoba_regressed.toFixed(3)}</span>
      </div>
      <div className={matchupStyles.statRow}>
        <span className={matchupStyles.statLabel}>K% / BB%</span>
        <span className={`fbq-num ${matchupStyles.statValue}`}>
          {(tte.metrics.k_pct * 100).toFixed(1)}% / {(tte.metrics.bb_pct * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

function LineupList({
  team,
  lineup,
  tte,
}: {
  team: string;
  lineup: LineupPlayer[];
  tte: TrueTalentAdjustment | undefined;
}) {
  return (
    <div className={matchupStyles.card}>
      <div className={matchupStyles.teamLabel}>{team}</div>
      <TeamOffenseAggregate tte={tte} />
      <div className={styles.divider} />
      {lineup.length === 0 ? (
        <div className={matchupStyles.emptyNote}>MLB todavía no ha publicado el lineup para este juego.</div>
      ) : (
        <div className={styles.list}>
          {lineup.map((p, i) => (
            <div key={p.id} className={styles.row}>
              <span className={styles.idx}>{i + 1}</span>
              <PlayerAvatar playerId={p.id} name={p.name} size={26} />
              <span className={matchupStyles.playerName}>{p.name}</span>
              {p.position && <span className={styles.pos}>{p.position}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ProjectedLineup({ schedule, prediction }: Props) {
  const { tte_away, tte_home } = prediction.metadata;
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
        <span className={matchupStyles.emptyNote} style={{ marginLeft: 8 }}>
          Métricas por bateador no están expuestas por el pipeline todavía — solo el agregado de equipo (TTE) de
          abajo.
        </span>
      </div>
      <div className={matchupStyles.grid2}>
        <LineupList team={schedule.away_team} lineup={schedule.away_lineup} tte={tte_away} />
        <LineupList team={schedule.home_team} lineup={schedule.home_lineup} tte={tte_home} />
      </div>
    </div>
  );
}
