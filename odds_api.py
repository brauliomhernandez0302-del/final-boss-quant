import requests
import requests

# 🔑 Tu API key personal de The Odds API
API_KEY = "4198abca7014f3c2c2a8ad036209e15b"

# 🌐 URL base de la API
BASE_URL = "https://api.the-odds-api.com/v4"


def get_odds(sport="soccer_epl", region="us", market="h2h"):
    """
    Obtiene las cuotas actuales para un deporte (por defecto Premier League)
    """
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
