import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import App from "../App";
import {
  matchupNoMoneyline,
  matchupWithPinnacleMoneyline,
} from "../test-fixtures/matchup_with_moneyline";
import type { ScheduleGame } from "../api/types";

const GAMES: ScheduleGame[] = [
  {
    game_pk: 823519,
    home_team: "New York Yankees",
    away_team: "Pittsburgh Pirates",
    home_team_id: 147,
    away_team_id: 134,
    venue: "Yankee Stadium",
    status: "Scheduled",
    game_date: "2026-07-21T23:05:00Z",
    official_date: "2026-07-21",
  },
];

function mockFetchFor(matchupPayload: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      if (url.includes("/api/mlb/games")) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(GAMES) });
      }
      if (url.includes("/api/mlb/matchup/")) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(matchupPayload) });
      }
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    }),
  );
}

describe("Matchup dashboard — real data shape (game_pk 823519, no moneyline market)", () => {
  beforeEach(() => mockFetchFor(matchupNoMoneyline));
  afterEach(() => vi.unstubAllGlobals());

  it("renders both teams, lambdas, and the honest 'no moneyline' note without crashing", async () => {
    render(<App />);

    // Both team names legitimately repeat across sections (header, prob bar,
    // lambda card, bullpen card) — assert presence, not uniqueness.
    expect((await screen.findAllByText("New York Yankees")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("Pittsburgh Pirates").length).toBeGreaterThan(0);

    // Real pipeline output for this game had no moneyline market at all
    // (metadata.value was entirely absent) — must degrade honestly, not crash.
    expect(
      await screen.findByText(/Sin mercado moneyline disponible/i),
    ).toBeInTheDocument();

    // Lineup wasn't posted yet for this real game — must show UNCONFIRMED,
    // never a fabricated batting order.
    expect(await screen.findAllByText(/UNCONFIRMED/i)).not.toHaveLength(0);

    // This capture predates `bullpen_usage`, so the bullpen card must say so
    // instead of rendering the old "roster activo" list. That list is exactly
    // what VAL-7.2 flagged: it used to assert "Paul Skenes" — the Pirates' ACE
    // STARTER — as proof the bullpen rendered, which is the defect (rotation
    // starters, and later a catcher with one mop-up inning, listed as the
    // bullpen the engine used). Asserting the honest fallback instead.
    expect(await screen.findAllByText(/bullpen_usage/i)).toHaveLength(2); // una nota por equipo
    expect(screen.queryByText("Paul Skenes")).not.toBeInTheDocument();
  });
});

describe("Matchup dashboard — synthetic Pinnacle moneyline fixture", () => {
  beforeEach(() => mockFetchFor(matchupWithPinnacleMoneyline));
  afterEach(() => vi.unstubAllGlobals());

  it("labels the big number as market-adjusted (Platt-2D) and shows the Platt-1D model number alongside it", async () => {
    render(<App />);

    expect(await screen.findByText(/AJUSTADA MERCADO/i)).toBeInTheDocument();
    expect(screen.getByText(/Platt-2D/)).toBeInTheDocument();
    // Platt-1D model probability must still be visible, not silently dropped.
    expect(screen.getByText(/Modelo \(Platt-1D\)/i)).toBeInTheDocument();

    // Value bet row rendered with a tier badge, not a blank/crashed row.
    expect(await screen.findByText("A")).toBeInTheDocument();
  });

  it("never renders a raw unrounded float for the headline probability", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText(/AJUSTADA MERCADO/i)).toBeInTheDocument());
    // 0.612 rounds to 61.2% — the raw 8-decimal-style float must never appear.
    expect(screen.queryByText(/0\.612000/)).not.toBeInTheDocument();
  });
});
