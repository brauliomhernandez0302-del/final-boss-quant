import type { Prediction } from "../../api/types";
import { JsonTree } from "./JsonTree";
import styles from "./TechnicalDetails.module.css";

interface Props {
  prediction: Prediction;
}

const ENGINE_LABELS: Record<string, string> = {
  tte_home: "True Talent Engine — home",
  tte_away: "True Talent Engine — away",
  pitcher: "Pitcher Engine",
  contextual: "Contextual Engine (rest/B2B)",
  bullpen: "Bullpen Engine",
  park_weather: "Park + Weather Engine",
  defense: "Defensive Efficiency Engine",
  hfa: "HFA + Travel Engine",
  market_odds: "Market odds (raw)",
  value: "Value Detector (full)",
};

export function TechnicalDetails({ prediction }: Props) {
  const { metadata, lambdas_history } = prediction;

  return (
    <div className={styles.wrap}>
      <div className={styles.block}>
        <div className={styles.blockTitle}>λ por etapa del pipeline</div>
        <JsonTree value={lambdas_history} />
      </div>
      {Object.entries(metadata).map(([key, value]) => (
        <div className={styles.block} key={key}>
          <div className={styles.blockTitle}>{ENGINE_LABELS[key] ?? key}</div>
          <JsonTree value={value} />
        </div>
      ))}
    </div>
  );
}
