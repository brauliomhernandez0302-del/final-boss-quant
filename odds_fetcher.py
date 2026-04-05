# ============================================================
# 🧠 SISTEMA DE APUESTAS INTELIGENTE
# odds_fetcher.py · G11 ULTRA REAL MODE (MEJORADO)
# FixOdds + SmartCache + Multi-Region + Error Handling
# ============================================================
# Braulio + GPT-5 · 2025
# ============================================================

import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

# ==========================
# CONFIGURACIÓN GLOBAL
# ==========================
load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
BASE_URL = "https://api.the-odds-api.com/v4"

# Regiones en orden de prioridad (MULTI-REGION para tier PRO)
REGIONS = ["us", "uk", "eu"]  # 3 regiones = mejor cobertura

# Mercados a consultar (COMPLETO para tier PRO)
MARKETS = ["h2h", "totals", "spreads"]  # Moneyline + O/U + Handicap

# Deportes soportados (optimizado para reducir llamadas API)
SPORTS_KEYS = [
    # Baseball
    "baseball_mlb",
    
    # Basketball
    "basketball_nba",
    "basketball_ncaab",
    "basketball_euroleague",
    
    # Soccer
    "soccer_epl",              # Premier League
    "soccer_spain_la_liga",    # La Liga
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_usa_mls",
    
    # MMA/UFC
    "mma_mixed_martial_arts",
    
    # American Football
    "americanfootball_nfl",
    "americanfootball_ncaaf",
    
    # Hockey
    "icehockey_nhl",
]

# Configuración de caché
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "odds_last.json"
CACHE_TTL_SECONDS = 10 * 60  # 10 minutos (reducido para datos más frescos)

# ==========================
# UTILIDADES
# ==========================

def save_cache(data: List[Dict[str, Any]]) -> None:
    """Guarda datos en caché con timestamp."""
    try:
        cache_data = {
            "timestamp": time.time(),
            "timestamp_readable": datetime.now().isoformat(),
            "total_events": len(data),
            "data": data
        }
        CACHE_FILE.write_text(
            json.dumps(cache_data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        print(f"💾 Caché guardado: {len(data)} eventos")
    except Exception as e:
        print(f"⚠️ Error guardando caché: {e}")


def load_cache() -> Optional[List[Dict[str, Any]]]:
    """Carga datos desde caché si es válido."""
    if not CACHE_FILE.exists():
        return None
    
    try:
        cache_content = json.loads(CACHE_FILE.read_text(encoding='utf-8'))
        timestamp = cache_content.get("timestamp", 0)
        
        # Verificar si el caché aún es válido
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            age_minutes = (time.time() - timestamp) / 60
            print(f"♻️ Usando caché ({age_minutes:.1f} min de antigüedad)")
            return cache_content.get("data", [])
        else:
            print(f"⏰ Caché expirado (>{CACHE_TTL_SECONDS//60} min)")
            return None
    except Exception as e:
        print(f"⚠️ Error leyendo caché: {e}")
        return None


def get_with_retries(url: str, max_retries: int = 3, delay: int = 2) -> Optional[Dict]:
    """Realiza petición HTTP con reintentos."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=15)
            
            # Éxito
            if response.status_code == 200:
                return response.json()
            
            # Rate limit
            elif response.status_code == 429:
                wait_time = int(response.headers.get('Retry-After', delay * 2))
                print(f"⏳ Rate limit - esperando {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            # Error de API
            elif response.status_code == 401:
                print(f"❌ API Key inválida o expirada")
                return None
            
            # Otros errores
            else:
                print(f"⚠️ Intento {attempt + 1}/{max_retries} - Status: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"⏱️ Timeout en intento {attempt + 1}/{max_retries}")
        
        except requests.exceptions.ConnectionError:
            print(f"🔌 Error de conexión en intento {attempt + 1}/{max_retries}")
        
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
        
        # Esperar antes del siguiente intento
        if attempt < max_retries - 1:
            time.sleep(delay * (attempt + 1))
    
    return None


def normalize_odds(odds_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliza y limpia los datos de odds."""
    normalized = {
        "sport_key": odds_data.get("sport_key", ""),
        "sport_title": odds_data.get("sport_title", ""),
        "home_team": odds_data.get("home_team", ""),
        "away_team": odds_data.get("away_team", ""),
        "commence_time": odds_data.get("commence_time", ""),
        "home_odds": None,
        "draw_odds": None,
        "away_odds": None,
        "total_line": None,
        "over_odds": None,
        "under_odds": None,
        "bookmaker": None,
        "last_update": datetime.now().isoformat()
    }
    
    try:
        bookmakers = odds_data.get("bookmakers", [])
        
        if not bookmakers:
            return normalized
        
        # Usar el primer bookmaker disponible (o implementar lógica de selección)
        best_bookmaker = bookmakers[0]
        normalized["bookmaker"] = best_bookmaker.get("title", "Unknown")
        
        # Extraer markets
        for market in best_bookmaker.get("markets", []):
            market_key = market.get("key", "")
            outcomes = market.get("outcomes", [])
            
            # Head to Head (Moneyline)
            if market_key == "h2h":
                for outcome in outcomes:
                    name = outcome.get("name", "").strip()
                    price = outcome.get("price")
                    
                    if name == normalized["home_team"]:
                        normalized["home_odds"] = price
                    elif name == normalized["away_team"]:
                        normalized["away_odds"] = price
                    elif name.lower() in ["draw", "empate", "tie"]:
                        normalized["draw_odds"] = price
            
            # Totals (Over/Under)
            elif market_key == "totals":
                if outcomes:
                    # La línea total es la misma para Over y Under
                    normalized["total_line"] = outcomes[0].get("point")
                    
                    for outcome in outcomes:
                        outcome_name = outcome.get("name", "").lower()
                        price = outcome.get("price")
                        
                        if outcome_name == "over":
                            normalized["over_odds"] = price
                        elif outcome_name == "under":
                            normalized["under_odds"] = price
    
    except Exception as e:
        print(f"⚠️ Error normalizando datos de {normalized.get('home_team', 'unknown')}: {e}")
    
    return normalized


# ==========================
# FUNCIÓN PRINCIPAL
# ==========================

def fetch_all_sports() -> List[Dict[str, Any]]:
    """
    Obtiene odds de todos los deportes configurados.
    Retorna lista de eventos normalizados.
    """
    all_data = []
    api_calls = 0
    max_calls = 50  # Límite de seguridad
    
    for sport_key in SPORTS_KEYS:
        if api_calls >= max_calls:
            print(f"⚠️ Límite de {max_calls} llamadas alcanzado - deteniendo")
            break
        
        # Solo usar región US para reducir llamadas
        region = "us"
        
        # Construir URL con ambos mercados
        url = (
            f"{BASE_URL}/sports/{sport_key}/odds/"
            f"?apiKey={ODDS_API_KEY}"
            f"&regions={region}"
            f"&markets={','.join(MARKETS)}"
            f"&oddsFormat=decimal"
        )
        
        print(f"\n🔍 Consultando {sport_key} ({region.upper()})...")
        
        data = get_with_retries(url)
        api_calls += 1
        
        if not data:
            print(f"❌ Sin datos para {sport_key}")
            continue
        
        # Normalizar cada evento
        count = 0
        for event in data:
            normalized_event = normalize_odds(event)
            
            # Solo agregar si tiene odds válidas
            if normalized_event.get("home_odds") and normalized_event.get("away_odds"):
                all_data.append(normalized_event)
                count += 1
        
        print(f"✅ {sport_key}: {count} eventos agregados")
        
        # Pequeño delay entre llamadas para evitar rate limit
        time.sleep(0.5)
    
    # Guardar en caché
    if all_data:
        save_cache(all_data)
        print(f"\n💾 Total: {len(all_data)} eventos guardados en caché")
    
    return all_data


def get_odds_data() -> List[Dict[str, Any]]:
    """
    Función principal para obtener odds (con caché inteligente).
    Esta es la función que llama app.py
    """
    # Verificar API key
    if not ODDS_API_KEY:
        print("❌ ERROR: ODDS_API_KEY no configurada en .env")
        print("💡 Crea un archivo .env con: ODDS_API_KEY=tu_key_aqui")
        
        # Intentar cargar desde caché aunque esté expirado
        if CACHE_FILE.exists():
            try:
                cache = json.loads(CACHE_FILE.read_text(encoding='utf-8'))
                print("⚠️ Usando caché antiguo (sin API key)")
                return cache.get("data", [])
            except Exception:
                pass
        
        return []
    
    # Intentar cargar desde caché válido
    cached_data = load_cache()
    if cached_data:
        return cached_data
    
    # Si no hay caché válido, fetch nuevo
    print("🌐 Obteniendo datos frescos de The Odds API...")
    return fetch_all_sports()


def get_sport_summary(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Genera resumen por deporte."""
    summary = {}
    for event in data:
        sport = event.get("sport_key", "unknown")
        summary[sport] = summary.get(sport, 0) + 1
    return summary


# ==========================
# EJECUCIÓN PRINCIPAL (TESTING)
# ==========================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ODDS FETCHER G11 ULTRA - TEST MODE")
    print("=" * 60)
    
    # Verificar configuración
    if not ODDS_API_KEY:
        print("\n❌ ERROR: No se encontró ODDS_API_KEY en .env")
        print("\n📝 Pasos para configurar:")
        print("1. Crea archivo .env en la raíz del proyecto")
        print("2. Agrega: ODDS_API_KEY=tu_key_de_the_odds_api")
        print("3. Obtén tu key en: https://the-odds-api.com/")
        exit(1)
    
    print(f"\n✅ API Key configurada: {ODDS_API_KEY[:8]}...{ODDS_API_KEY[-4:]}")
    print(f"📦 Caché TTL: {CACHE_TTL_SECONDS // 60} minutos")
    print(f"🎯 Deportes configurados: {len(SPORTS_KEYS)}")
    
    # Obtener datos
    print("\n" + "=" * 60)
    odds_data = get_odds_data()
    print("=" * 60)
    
    if not odds_data:
        print("\n❌ No se obtuvieron datos")
        print("💡 Verifica tu API key y conexión a internet")
        exit(1)
    
    # Mostrar resumen
    print(f"\n📊 RESUMEN:")
    print(f"   Total eventos: {len(odds_data)}")
    
    summary = get_sport_summary(odds_data)
    print(f"\n📋 Por deporte:")
    for sport, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {sport}: {count} eventos")
    
    # Mostrar ejemplo
    if odds_data:
        print(f"\n🎯 EJEMPLO DE EVENTO:")
        example = odds_data[0]
        print(f"   Deporte: {example.get('sport_title')}")
        print(f"   Partido: {example.get('home_team')} vs {example.get('away_team')}")
        print(f"   Cuotas: {example.get('home_odds')} / {example.get('draw_odds', 'N/A')} / {example.get('away_odds')}")
        if example.get('total_line'):
            print(f"   Total: {example.get('total_line')} (Over: {example.get('over_odds')}, Under: {example.get('under_odds')})")
        print(f"   Fecha: {example.get('commence_time')}")
    
    print("\n✅ Fetch completado - G11 Ultra Real Mode activo")
    print("=" * 60)
