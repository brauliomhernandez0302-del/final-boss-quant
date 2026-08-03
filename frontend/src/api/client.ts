// Thin fetch wrapper over api/server.py — the ONLY place the frontend's
// base URL is configured, so future screens (track record, CLV report,
// status, config) reuse this instead of each hardcoding fetch() calls.
import type { MatchupPayload, ScheduleGame } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:5000";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  games: () => getJson<ScheduleGame[]>("/api/mlb/games"),
  matchup: (gamePk: number) => getJson<MatchupPayload>(`/api/mlb/matchup/${gamePk}`),
};
