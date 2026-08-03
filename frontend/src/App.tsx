import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { MatchupPage } from "./pages/MatchupPage";
import { ComingSoonPage } from "./pages/ComingSoonPage";

// Router scaffolding for the phases after this one: track record / CLV /
// status / config each mount here later as their own <Route>, sharing this
// same AppShell (theme + nav) and the api/client.ts data layer. Only the
// matchup dashboard is a real page today.
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<MatchupPage />} />
          <Route path="track-record" element={<ComingSoonPage title="Track Record" />} />
          <Route path="clv" element={<ComingSoonPage title="CLV Report" />} />
          <Route path="status" element={<ComingSoonPage title="Status" />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
