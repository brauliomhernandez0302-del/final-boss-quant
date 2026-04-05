import requests
import pandas as pd

# 🔑 Tu API Key personal de The Odds API
API_KEY = "4198abca7014f3c2c2a83ad36299e15b"

# 🌍 URL base de la API
BASE_URL = "https://api.the-odds-api.com/v4/sports"


# ⚽ Función para obtener cuotas por deporte
def get_odds(sport="soccer_epl", region="us", market="h2h"):
    url = f"{BASE_URL}/{sport}/odds/?apiKey={API_KEY}&regions={region}&markets={market}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print("❌ Error:", response.status_code, response.text)
            return None
        return response.json()
    except Exception as e:
        print("⚠️ Error en la conexión:", e)
        return None


# 📊 Función para formatear los datos de cuotas
def format_odds(odds_data):
    rows = []
    for game in odds_data:
        home = game.get("home_team", "Desconocido")
        away = game.get("away_team", "Desconocido")
        for bookmaker in game.get("bookmakers", []):
            name = bookmaker.get("title", "N/A")
            markets = bookmaker.get("markets", [])
            if not markets:
                continue
            outcomes = markets[0].get("outcomes", [])
            for m in outcomes:
                rows.append({
                    "bookmaker": name,
                    "home_team": home,
                    "away_team": away,
                    "selection": m.get("name", "N/A"),
                    "price": m.get("price", "N/A")
                })
    return pd.DataFrame(rows)


# 💰 Función para buscar la mejor cuota por selección
def best_odds(df):
    return df.groupby(["home_team", "away_team", "selection"])["price"].max().reset_index()


# 🚀 Ejecución principal
if __name__ == "__main__":
    print("🔄 Obteniendo cuotas de The Odds API...\n")
    odds_data = get_odds()

    if odds_data:
        print("✅ Conexión exitosa, mostrando primeras cuotas...\n")
        df = format_odds(odds_data)
        print(df.head(10))

        print("\n🏆 Mejores cuotas disponibles por selección:\n")
        best = best_odds(df)
        print(best)

        best.to_excel("cuotas_actuales.xlsx", index=False)
        print("\n📁 Archivo guardado como 'cuotas_actuales.xlsx'")
    else:
        print("❌ No se pudieron obtener las cuotas.")
