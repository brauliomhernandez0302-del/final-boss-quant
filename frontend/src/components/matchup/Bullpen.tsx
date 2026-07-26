import { PlayerAvatar } from "../shared/PlayerAvatar";
import { ProvenanceBadge } from "../shared/ProvenanceBadge";
import { DirectionalBar } from "../shared/DirectionalBar";
import { DuelBar } from "../shared/DuelBar";
import type { BullpenUsage, MatchupPayload } from "../../api/types";
import matchupStyles from "./matchup.module.css";
import styles from "./Bullpen.module.css";

interface Props {
  payload: MatchupPayload;
}

export function Bullpen({ payload }: Props) {
  const { schedule, prediction, bullpen_usage } = payload;
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
            <span className={matchupStyles.emptyNote}>{away.tier_label}</span>
            <span className={matchupStyles.emptyNote}>{home.tier_label}</span>
          </div>
        </div>
      )}

      <div className={matchupStyles.emptyNote} style={{ margin: "12px 0" }}>
        Las listas de abajo son los relevistas que el engine <strong>realmente pesó</strong> en{" "}
        <code>total_mult</code>: su roster clasificado (cero aperturas esta temporada) filtrado a quienes tenían
        datos con los que pesarlos. Los números de cada fila SON el peso — PA de Savant para el agregado de
        xwOBA/barrel, IP para el de SIERA. Quien no aparece, no aportó nada al cálculo.
      </div>

      <div className={matchupStyles.grid2}>
        <UsageTable team={schedule.away_team} usage={bullpen_usage?.away} />
        <UsageTable team={schedule.home_team} usage={bullpen_usage?.home} />
      </div>
    </div>
  );
}

function UsageTable({ team, usage }: { team: string; usage?: BullpenUsage }) {
  if (!usage) {
    return (
      <div className={matchupStyles.card}>
        <div className={matchupStyles.teamLabel}>{team} — bullpen</div>
        <div className={matchupStyles.emptyNote}>
          Esta respuesta del API no trae <code>bullpen_usage</code>: es de antes de que el dashboard pudiera
          mostrar qué relevistas pesó el engine. La lista vieja ("roster activo") medía otra cosa, así que no se
          muestra.
        </div>
      </div>
    );
  }

  const notWeighted = usage.n_classified - usage.n_used;

  return (
    <div className={matchupStyles.card}>
      <div className={matchupStyles.teamLabel}>
        {team} — {usage.n_used} relevistas usados <ProvenanceBadge>USADO POR EL ENGINE</ProvenanceBadge>
      </div>

      {usage.degraded ? (
        <div className={matchupStyles.emptyNote}>
          La clasificación de roles falló en esta llamada, así que el engine agregó sobre el roster completo. No se
          lista nada: cualquier lista de acá sería distinta de la que usó el modelo.
        </div>
      ) : usage.n_used === 0 ? (
        <div className={matchupStyles.emptyNote}>Ningún relevista de este roster tenía datos con los que pesarlo.</div>
      ) : (
        <>
          <div className={styles.table}>
            {usage.pitchers.map((p, i) => (
              <div key={p.id} className={styles.row}>
                <span className={styles.idx}>{i + 1}</span>
                <PlayerAvatar playerId={p.id} name={p.name} size={26} />
                <span className={styles.name}>
                  {p.name}
                  {p.position && p.position !== "P" && (
                    <span className={styles.positionFlag} title="Jugador de posición que lanzó en relevo">
                      {" "}
                      {p.position}
                    </span>
                  )}
                </span>
                <span className={`fbq-num ${styles.weights}`}>
                  {p.savant_pa != null ? `${p.savant_pa.toFixed(0)} PA` : "—"}
                  {" · "}
                  {p.siera_ip != null ? `${p.siera_ip.toFixed(1)} IP` : "—"}
                </span>
              </div>
            ))}
          </div>
          <div className={styles.excludedNote}>
            {usage.n_savant} con datos de Savant (xwOBA/barrel, 65% del quality) y {usage.n_siera} con SIERA/xFIP de
            FanGraphs (35%) — son las dos partes de esos {usage.n_used}, no otros conteos.
            {notWeighted > 0 &&
              (notWeighted === 1
                ? " 1 clasificado más quedó fuera: el engine no tenía dato con qué pesarlo."
                : ` ${notWeighted} clasificados más quedaron fuera: el engine no tenía datos con qué pesarlos.`)}
          </div>
        </>
      )}
    </div>
  );
}
