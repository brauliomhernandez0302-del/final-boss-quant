import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { MatchupPayload } from "../api/types";

interface State {
  data: MatchupPayload | null;
  loading: boolean;
  error: string | null;
}

export function useMatchup(gamePk: number | null): State {
  const [state, setState] = useState<State>({ data: null, loading: false, error: null });

  useEffect(() => {
    if (gamePk == null) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    let cancelled = false;
    setState({ data: null, loading: true, error: null });
    api
      .matchup(gamePk)
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null });
      })
      .catch((err: Error) => {
        if (!cancelled) setState({ data: null, loading: false, error: err.message });
      });
    return () => {
      cancelled = true;
    };
  }, [gamePk]);

  return state;
}
