import { PlayerAvatar } from "../shared/PlayerAvatar";
import { ProvenanceBadge } from "../shared/ProvenanceBadge";
import { DirectionalBar } from "../shared/DirectionalBar";
import { DuelBar } from "../shared/DuelBar";
import type { BullpenAdjustment, MatchupPayload, RosterPitcher } from "../../api/types";
import matchupStyles from "./matchup.module.css";
import styles from "./Bullpen.module.css";

interface Props {
  payload: MatchupPayload;
}

export function Bullpen({ payload }: Props) {
  const { schedule, prediction, bullpen_roster } = payload;
  const bullpenMeta = prediction.metadata.bullpen;
  const away = bullpenMeta?.bullpen_away;
  const home = bullpenMeta?.bullpen_home;

  return (
    <div>
      {away && home && (
        <div className={styles.aggregate}>
          <div className={styles.aggregateHeader}>
            <span>{schedule.away_team}</span>
            <span className={matchupStyles.emptyNote}>agregado del bullpen</span>
            <span>{schedule.home_team}</span>
          </div>
          <DuelBar
            label={away.used_siera && home.used_siera ? "SIERA" : "ERA"}
            awayValue={away.used_siera ? away.siera_ag! : away.era}
            homeValue={home.used_siera ? home.siera_ag! : home.era}
            formatValue={(v) => v.toFixed(2)}
            lowerIsBetter
            domainHalf={1.5}
            center={4}
          />
          <DuelBar
            label="ERA efectiva"
            awayValue={away.effective_era}
            homeValue={home.effective_era}
            formatValue={(v) => v.toFixed(2)}
            lowerIsBetter
            domainHalf={1.5}
            center={4}
          />
          <div className={styles.multRow}>
            <div>
              <DirectionalBar
                label="workload_mult"
                value={away.workload_mult}
                domainHalf={0.1}
                formatValue={(v) => v.toFixed(3)}
              />
              <DirectionalBar
                label="total_mult"
                value={away.total_mult}
                domainHalf={0.1}
                formatValue={(v) => v.toFixed(3)}
              />
            </div>
            <div>
              <DirectionalBar
                label="workload_mult"
                value={home.workload_mult}
                domainHalf={0.1}
                formatValue={(v) => v.toFixed(3)}
              />
              <DirectionalBar
                label="total_mult"
                value={home.total_mult}
                domainHalf={0.1}
                formatValue={(v) => v.toFixed(3)}
              />
            </div>
          </div>
          <div className={styles.tierRow}>
            <span className={matchupStyles.emptyNote}>
              {away.tier_label} · {away.n_pitchers} disponibles (pipeline)
            </span>
            <span className={matchupStyles.emptyNote}>
              {home.tier_label} · {home.n_pitchers} disponibles (pipeline)
            </span>
          </div>
        </div>
      )}

      <div className={matchupStyles.emptyNote} style={{ margin: "12px 0" }}>
        El pipeline solo calcula estadísticas de bullpen a nivel de EQUIPO (arriba) — el API no expone SIERA/ERA por
        relevista individual todavía. El roster abajo es identidad, no ranking.
      </div>

      <div className={matchupStyles.grid2}>
        <RosterTable team={schedule.away_team} roster={bullpen_roster.away} meta={away} />
        <RosterTable team={schedule.home_team} roster={bullpen_roster.home} meta={home} />
      </div>
    </div>
  );
}

function RosterTable({
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
      <div className={matchupStyles.teamLabel}>
        {team} — roster activo <ProvenanceBadge>ROSTER — NO CONFIRMADO</ProvenanceBadge>
      </div>
      {roster.length === 0 ? (
        <div className={matchupStyles.emptyNote}>Sin datos de roster disponibles.</div>
      ) : (
        <div className={styles.table}>
          {roster.map((p, i) => (
            <div key={p.id} className={styles.row}>
              <span className={styles.idx}>{i + 1}</span>
              <PlayerAvatar playerId={p.id} name={p.name} size={26} />
              <span className={styles.name}>{p.name}</span>
            </div>
          ))}
        </div>
      )}
      {meta && (
        <div className={styles.excludedNote}>
          Excluye al abridor probable — {meta.n_siera_pitchers ?? meta.n_pitchers} entraron al cálculo del pipeline.
        </div>
      )}
    </div>
  );
}
