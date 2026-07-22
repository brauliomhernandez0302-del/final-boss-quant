import { Section } from "../shared/Section";
import { MatchupHeader } from "./MatchupHeader";
import { CoreMetrics } from "./CoreMetrics";
import { StartingPitchers } from "./StartingPitchers";
import { Bullpen } from "./Bullpen";
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
      <Section title="Core Metrics">
        <CoreMetrics prediction={payload.prediction} schedule={payload.schedule} />
      </Section>
      <Section title="◆ Starting Pitchers">
        <StartingPitchers payload={payload} />
      </Section>
      <Section title="◆ Bullpen">
        <Bullpen payload={payload} />
      </Section>
      <Section title="◆ Projected Lineup">
        <ProjectedLineup schedule={payload.schedule} />
      </Section>
      <Section title="◆ Value Bets · Pinnacle Ref">
        <ValueBets prediction={payload.prediction} />
      </Section>
      <Section title="Technical Details" collapsible defaultOpen={false}>
        <TechnicalDetails prediction={payload.prediction} />
      </Section>
    </div>
  );
}
