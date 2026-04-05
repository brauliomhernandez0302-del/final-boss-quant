from __future__ import annotations
import os
import requests
from dotenv import load_dotenv

# Cargar variables desde .env
load_dotenv()

# Configuración base
BASE = os.getenv("ODDS_API_BASE", "https://api.the-odds-api.com/v4")
KEY = os.getenv("ODDS_API_KEY", "")
DEFAULT_BOOKS = ["pinnacle", "betonlineag", "draftkings", "fanduel"]


class OddsClient:
    """
    Cliente universal para consultar cuotas desde The Odds API.
    Si no hay clave en .env, entra automáticamente en modo DEMO.
    """

    def __init__(self, api_key: str | None = None):
        self.key = api_key or KEY

    def _get(self, path: str, params: dict | None = None):
        """Método interno para peticiones GET con manejo de demo."""
        if not self.key:
            print("⚠️ Modo DEMO activo: no se detectó API key, retornando datos simulados.")
            return {"demo": True, "data": []}

        params = dict(params or {})
        params["apiKey"] = self.key
        url = f"{BASE.rstrip('/')}/{path.lstrip('/')}"

        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"❌ Error al realizar solicitud a The Odds API: {e}")
            return {"error": str(e), "data": []}

    # --- Endpoints de ejemplo ---
    def soccer_totals(
        self,
        regions: str = "us,eu",
        markets: str = "totals,bt_both_teams_to_score",
        oddsFormat: str = "american",
    ):
        """Obtiene cuotas de fútbol (EPL por defecto)."""
        return self._get(
            "sports/soccer_epl/odds",
            {
                "regions": regions,
                "markets": markets,
                "oddsFormat": oddsFormat,
                "bookmakers": ",".join(DEFAULT_BOOKS),
            },
        )

    def basketball_totals(
        self,
        sport_key: str = "basketball_nba",
        regions: str = "us,eu",
        markets: str = "totals,spreads",
    ):
        """Obtiene cuotas NBA."""
        return self._get(
            f"sports/{sport_key}/odds",
            {
                "regions": regions,
                "markets": markets,
                "oddsFormat": "american",
                "bookmakers": ",".join(DEFAULT_BOOKS),
            },
        )

    def baseball_totals(
        self,
        sport_key: str = "baseball_mlb",
        regions: str = "us,eu",
        markets: str = "totals",
    ):
        """Obtiene cuotas MLB."""
        return self._get(
            f"sports/{sport_key}/odds",
            {
                "regions": regions,
                "markets": markets,
                "oddsFormat": "american",
                "bookmakers": ",".join(DEFAULT_BOOKS),
            },
        )

