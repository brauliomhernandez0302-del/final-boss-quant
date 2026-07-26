import { PlayerAvatar } from "../shared/PlayerAvatar";
import { DuelBar } from "../shared/DuelBar";
import type { MatchupPayload, PitcherAdjustment } from "../../api/types";
import matchupStyles from "./matchup.module.css";
import styles from "./StartingPitchers.module.css";

interface Props {
  payload: MatchupPayload;
}

const PCT_FMT = (v: number) => v.toFixed(3);

export function StartingPitchers({ payload }: Props) {
  const { schedule, prediction, pitcher_bio } = payload;
  const pitcherMeta = prediction.metadata.pitcher;
  const away = pitcherMeta?.pitcher_away;
  const home = pitcherMeta?.pitcher_home;

  return (
    <div className={styles.wrap}>
      <div className={styles.header}>
        <div className={styles.playerCol}>
          <PlayerAvatar playerId={schedule.away_pitcher_id} name={schedule.away_pitcher ?? "TBD"} size={52} />
          <div>
            <div className={matchupStyles.playerName}>{schedule.away_pitcher ?? "TBD"}</div>
            <div className={matchupStyles.playerMeta}>
              {schedule.away_team}
              {pitcher_bio.away?.throws ? ` · ${pitcher_bio.away.throws}HP` : ""}
            </div>
          </div>
        </div>
        <span className={styles.vs}>VS</span>
        <div className={`${styles.playerCol} ${styles.playerColRight}`}>
          <div className={styles.textRight}>
            <div className={matchupStyles.playerName}>{schedule.home_pitcher ?? "TBD"}</div>
            <div className={matchupStyles.playerMeta}>
              {schedule.home_team}
              {pitcher_bio.home?.throws ? ` · ${pitcher_bio.home.throws}HP` : ""}
            </div>
          </div>
          <PlayerAvatar playerId={schedule.home_pitcher_id} name={schedule.home_pitcher ?? "TBD"} size={52} />
        </div>
      </div>

      {away && home ? (
        <div className={styles.rows}>
          <DuelRow label="quality_mult" away={away} home={home} field="quality_mult" />
          <DuelRow label="form_mult" away={away} home={home} field="form_mult" />
          <DuelRow label="matchup_mult" away={away} home={home} field="matchup_mult" />
          <DuelRow label="platoon_mult" away={away} home={home} field="platoon_mult" />
          <DuelRow label="fatigue_mult" away={away} home={home} field="fatigue_mult" />
          <div className={styles.totalDivider} />
          <DuelRow label="total_multiplier" away={away} home={home} field="total_multiplier" emphasize />
        </div>
      ) : (
        <div className={matchupStyles.emptyNote} style={{ marginTop: 12 }}>
          Sin ajuste de Pitcher Engine disponible para este juego.
        </div>
      )}
      <div className={matchupStyles.emptyNote} style={{ marginTop: 8 }}>
        Menor multiplicador = pitcher suprime más carreras del rival — no es un juicio de "bueno/malo" fuera de ese
        eje.
      </div>
    </div>
  );
}

function DuelRow({
  label,
  away,
  home,
  field,
  emphasize,
}: {
  label: string;
  away: PitcherAdjustment;
  home: PitcherAdjustment;
  field: keyof PitcherAdjustment;
  emphasize?: boolean;
}) {
  return (
    <div className={emphasize ? styles.emphasizedRow : undefined}>
      <DuelBar
        label={label}
        awayValue={away[field] as number}
        homeValue={home[field] as number}
        formatValue={PCT_FMT}
        lowerIsBetter
        domainHalf={0.2}
      />
    </div>
  );
}
