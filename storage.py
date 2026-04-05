from pathlib import Path
import pandas as pd

# Ruta donde se guardará el registro de apuestas
LOG_FILE = Path("bets_log.csv")


def save_bets(df: pd.DataFrame):
    """
    Guarda los picks seleccionados en bets_log.csv.
    Si ya existe, añade las nuevas apuestas sin borrar las anteriores.
    """
    if df is None or df.empty:
        print("⚠️ No hay apuestas para guardar.")
        return

    try:
        if LOG_FILE.exists():
            prev = pd.read_csv(LOG_FILE)
            out = pd.concat([prev, df], ignore_index=True)
        else:
            out = df.copy()

        out.to_csv(LOG_FILE, index=False)
        print(f"✅ {len(df)} apuestas guardadas en {LOG_FILE}")

    except Exception as e:
        print(f"❌ Error al guardar las apuestas: {e}")


def load_bets() -> pd.DataFrame:
    """
    Carga el historial de apuestas desde bets_log.csv.
    Si no existe, devuelve un DataFrame vacío.
    """
    try:
        if LOG_FILE.exists():
            df = pd.read_csv(LOG_FILE)
            print(f"📂 Historial cargado ({len(df)} registros).")
            return df
        else:
            print("ℹ️ No existe un archivo de historial aún.")
            return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Error al cargar historial: {e}")
        return pd.DataFrame()
