import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ScheduleGame } from "../api/types";

interface State {
  games: ScheduleGame[];
  loading: boolean;
  error: string | null;
}

export function useGamesList(): State {
  const [state, setState] = useState<State>({ games: [], loading: true, error: null });

  useEffect(() => {
    let cancelled = false;
    api
      .games()
      .then((games) => {
        if (!cancelled) setState({ games, loading: false, error: null });
      })
      .catch((err: Error) => {
        if (!cancelled) setState({ games: [], loading: false, error: err.message });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
