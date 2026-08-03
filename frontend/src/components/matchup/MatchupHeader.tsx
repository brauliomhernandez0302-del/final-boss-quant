import { TeamLogo } from "../shared/TeamLogo";
import type { ScheduleDetail } from "../../api/types";
import styles from "./MatchupHeader.module.css";

interface Props {
  schedule: ScheduleDetail;
}

export function MatchupHeader({ schedule }: Props) {
  return (
    <header className={styles.header}>
      <div className={styles.matchup}>
        <div className={styles.teamBlock}>
          <div className={styles.logoFrame}>
            <TeamLogo teamId={schedule.away_team_id} teamName={schedule.away_team} size={64} />
          </div>
          <span className={styles.teamName}>{schedule.away_team}</span>
          <span className={styles.roleTag}>AWAY</span>
        </div>
        <span className={styles.at}>@</span>
        <div className={styles.teamBlock}>
          <div className={styles.logoFrame}>
            <TeamLogo teamId={schedule.home_team_id} teamName={schedule.home_team} size={64} />
          </div>
          <span className={styles.teamName}>{schedule.home_team}</span>
          <span className={styles.roleTag}>HOME</span>
        </div>
      </div>

      <div className={styles.badges}>
        <span className={styles.liveBadge}>
          <span className={`${styles.liveDot} fbq-pulse-dot`} />
          LIVE MODEL
        </span>
        {schedule.venue && <span className={styles.chip}>{schedule.venue}</span>}
        {schedule.status && <span className={styles.chip}>{schedule.status}</span>}
        {schedule.is_playoff && <span className={styles.chip}>PLAYOFF</span>}
        <span className={`fbq-num ${styles.chip}`}>game_pk {schedule.game_pk}</span>
      </div>
    </header>
  );
}
