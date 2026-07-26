import { Section } from "../shared/Section";
import { MatchupHeader } from "./MatchupHeader";
import { CoreMetrics } from "./CoreMetrics";
import { LambdaWaterfall } from "./LambdaWaterfall";
import { StartingPitchers } from "./StartingPitchers";
import { Bullpen } from "./Bullpen";
import { GameContext } from "./GameContext";
import { ProjectedLineup } from "./ProjectedLineup";
import { ValueBets } from "./ValueBets";
import { TechnicalDetails } from "./TechnicalDetails";
import type { MatchupPayload } from "../../api/types";

interface Props {
  payload: MatchupPayload;
}

export function MatchupDashboard({ payload }: Props) {
  return (
    <div>
      <MatchupHeader schedule={payload.schedule} />
      <Section title="Core Metrics" index={0}>
        <CoreMetrics prediction={payload.prediction} schedule={payload.schedule} />
      </Section>
      <Section title="◆ Cascada de Factores λ" index={1}>
        <LambdaWaterfall prediction={payload.prediction} schedule={payload.schedule} />
      </Section>
      <Section title="◆ Abridores" index={2}>
        <StartingPitchers payload={payload} />
      </Section>
      <Section title="◆ Bullpen" index={3}>
        <Bullpen payload={payload} />
      </Section>
      <Section title="◆ Contexto de Juego" index={4}>
        <GameContext prediction={payload.prediction} schedule={payload.schedule} />
      </Section>
      <Section title="◆ Projected Lineup" index={5}>
        <ProjectedLineup schedule={payload.schedule} prediction={payload.prediction} />
      </Section>
      <Section title="◆ Value Bets · Pinnacle Ref" index={6}>
        <ValueBets prediction={payload.prediction} />
      </Section>
      <Section title="Technical Details" collapsible defaultOpen={false} index={7}>
        <TechnicalDetails prediction={payload.prediction} />
      </Section>
    </div>
  );
}
