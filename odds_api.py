import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY", "")
if not API_KEY:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "ODDS_API_KEY not set — get_best_odds_for_teams will return {}"
    )

BASE_URL = "https://api.the-odds-api.com/v4"


def get_odds(sport="soccer_epl", region="us", market="h2h"):
    """
    Obtiene las cuotas actuales para un deporte (por defecto Premier League)
    """
    if not API_KEY:
        return None
    url = f"{BASE_URL}/sports/{sport}/odds/?apiKey={API_KEY}&regions={region}&markets={market}"
    response = requests.get(url)

    if response.status_code != 200:
        print("❌ Error:", response.status_code, response.text)
        return None

    data = response.json()
    return data


if __name__ == "__main__":
    odds = get_odds()
    if odds:
        print("✅ Conexión exitosa. Mostrando algunos partidos:\n")
        for game in odds[:3]:  # muestra los 3 primeros
            print(game["home_team"], "vs", game["away_team"])
            for bookmaker in game["bookmakers"]:
                print(" -", bookmaker["title"], ":", bookmaker["markets"][0]["outcomes"])
    else:
        print("❌ No se pudieron obtener las cuotas.")
def get_best_odds_for_teams(home_team: str, away_team: str,
                             sport: str = "baseball_mlb",
                             region: str = "us") -> dict:
    """
    Busca las mejores odds para un partido por nombre de equipos.
    Retorna las odds más altas disponibles entre todos los bookmakers.
    """
    all_games = get_odds(sport=sport, region=region, market="h2h")

    if not all_games:
        return {}

    _PINNACLE_KEY = "pinnaclesports"

    for game in all_games:
        g_home = game.get("home_team", "")
        g_away = game.get("away_team", "")

        # fuzzy match
        if (home_team.lower() in g_home.lower() or g_home.lower() in home_team.lower()) and \
           (away_team.lower() in g_away.lower() or g_away.lower() in away_team.lower()):

            best_home = 0.0
            best_away = 0.0
            pin_home = None
            pin_away = None

            for bookmaker in game.get("bookmakers", []):
                is_pinnacle = bookmaker.get("key", "").lower() == _PINNACLE_KEY or \
                              "pinnacle" in bookmaker.get("title", "").lower()
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market.get("outcomes", []):
                            price = outcome["price"]
                            if outcome["name"] == g_home:
                                best_home = max(best_home, price)
                                if is_pinnacle:
                                    pin_home = price
                            elif outcome["name"] == g_away:
                                best_away = max(best_away, price)
                                if is_pinnacle:
                                    pin_away = price

            return {
                "home_team": g_home,
                "away_team": g_away,
                "ml_home": best_home if best_home > 0 else None,
                "ml_away": best_away if best_away > 0 else None,
                "pin_home": pin_home,
                "pin_away": pin_away,
                "game_id": game.get("id"),
            }

    return {}