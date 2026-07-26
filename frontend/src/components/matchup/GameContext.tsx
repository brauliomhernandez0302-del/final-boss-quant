import { DirectionalBar } from "../shared/DirectionalBar";
import { DuelBar } from "../shared/DuelBar";
import { ProvenanceBadge } from "../shared/ProvenanceBadge";
import type { Prediction, ScheduleDetail } from "../../api/types";
import matchupStyles from "./matchup.module.css";
import styles from "./GameContext.module.css";

interface Props {
  prediction: Prediction;
  schedule: ScheduleDetail;
}

export function GameContext({ prediction, schedule }: Props) {
  const { park_weather, defense, hfa, contextual } = prediction.metadata;

  return (
    <div className={styles.grid}>
      <div className={matchupStyles.card}>
        <div className={matchupStyles.teamLabel}>Parque &amp; clima</div>
        {park_weather ? (
          <>
            <div className={styles.parkLine}>
              <span className={styles.parkName}>{park_weather.park_name}</span>
              <span className={matchupStyles.emptyNote}>
                {park_weather.conditions}
                {park_weather.roof_closed ? " · techo cerrado" : ""}
              </span>
            </div>
            <DirectionalBar
              label="park_factor"
              value={park_weather.park_factor}
              domainHalf={0.15}
              formatValue={(v) => v.toFixed(3)}
            />
            {park_weather.roof_closed ? (
              <div className={matchupStyles.emptyNote} style={{ marginTop: 4 }}>
                Techo cerrado — clima no aplica (multiplicadores neutros).
              </div>
            ) : (
              <>
                {park_weather.temp_f != null && (
                  <DirectionalBar
                    label={`temp ${park_weather.temp_f.toFixed(1)}°F`}
                    value={park_weather.temp_mult}
                    domainHalf={0.05}
                    formatValue={(v) => v.toFixed(4)}
                  />
                )}
                {park_weather.wind_mph != null && (
                  <DirectionalBar
                    label={`viento ${park_weather.wind_mph.toFixed(1)}mph`}
                    value={park_weather.wind_mult}
                    domainHalf={0.05}
                    formatValue={(v) => v.toFixed(4)}
                  />
                )}
                <DirectionalBar
                  label="lluvia"
                  value={park_weather.rain_mult}
                  domainHalf={0.05}
                  formatValue={(v) => v.toFixed(4)}
                />
              </>
            )}
            {park_weather.weather_source === "missing" && (
              <div className={styles.warnRow}>
                <ProvenanceBadge>WEATHER SOURCE — MISSING</ProvenanceBadge>
              </div>
            )}
          </>
        ) : (
          <div className={matchupStyles.emptyNote}>Sin datos de park/weather engine.</div>
        )}
      </div>

      <div className={matchupStyles.card}>
        <div className={matchupStyles.teamLabel}>Defensa (DER / OAA)</div>
        {defense ? (
          <>
            <div className={styles.duelHeader}>
              <span>{schedule.away_team}</span>
              <span />
              <span>{schedule.home_team}</span>
            </div>
            <DuelBar
              label="DER regresado"
              awayValue={defense.away_defense.der_regressed}
              homeValue={defense.home_defense.der_regressed}
              formatValue={(v) => v.toFixed(4)}
              lowerIsBetter={false}
              domainHalf={0.03}
              center={0.7}
            />
            <DuelBar
              label="OAA"
              awayValue={defense.away_defense.oaa}
              homeValue={defense.home_defense.oaa}
              formatValue={(v) => v.toFixed(1)}
              lowerIsBetter={false}
              domainHalf={15}
              center={0}
            />
            <div className={matchupStyles.emptyNote} style={{ marginTop: 6 }}>
              El mult. de cada equipo aplica al bateo del RIVAL (juega defensa mientras el otro batea) — ya
              reflejado en la cascada λ de arriba.
            </div>
          </>
        ) : (
          <div className={matchupStyles.emptyNote}>Sin datos de defensive efficiency engine.</div>
        )}
      </div>

      <div className={matchupStyles.card}>
        <div className={matchupStyles.teamLabel}>Descanso / fatiga</div>
        {contextual ? (
          <>
            <div className={styles.restRow}>
              <span className={styles.restTeam}>{schedule.away_team}</span>
              <span className={matchupStyles.playerMeta}>
                {contextual.away_rest_days}d · {contextual.away_rest_reason}
              </span>
            </div>
            <DirectionalBar
              label="away_rest_mult"
              value={contextual.away_rest_mult}
              domainHalf={0.06}
              formatValue={(v) => v.toFixed(3)}
            />
            <div className={styles.restRow} style={{ marginTop: 10 }}>
              <span className={styles.restTeam}>{schedule.home_team}</span>
              <span className={matchupStyles.playerMeta}>
                {contextual.home_rest_days}d · {contextual.home_rest_reason}
              </span>
            </div>
            <DirectionalBar
              label="home_rest_mult"
              value={contextual.home_rest_mult}
              domainHalf={0.06}
              formatValue={(v) => v.toFixed(3)}
            />
          </>
        ) : (
          <div className={matchupStyles.emptyNote}>Sin datos de contextual engine.</div>
        )}
      </div>

      <div className={matchupStyles.card}>
        <div className={matchupStyles.teamLabel}>Home field advantage</div>
        {hfa ? (
          <>
            <DirectionalBar
              label="hfa_mult"
              value={hfa.hfa_mult}
              domainHalf={0.05}
              formatValue={(v) => v.toFixed(3)}
            />
            <div className={matchupStyles.statRow}>
              <span className={matchupStyles.statLabel}>uniform_home_mult</span>
              <span className={`fbq-num ${matchupStyles.statValue}`}>
                {hfa.uniform_home_mult.toFixed(3)}
              </span>
            </div>
            <div className={matchupStyles.statRow}>
              <span className={matchupStyles.statLabel}>fatiga de viaje ({schedule.away_team})</span>
              <span className={`fbq-num ${matchupStyles.statValue}`}>{hfa.travel_penalty.toFixed(3)}</span>
            </div>
            {hfa.travel_source === "missing" && (
              <div className={styles.warnRow}>
                <ProvenanceBadge>TRAVEL SOURCE — MISSING</ProvenanceBadge>
              </div>
            )}
          </>
        ) : (
          <div className={matchupStyles.emptyNote}>Sin datos de HFA engine.</div>
        )}
      </div>
    </div>
  );
}
