from dataclasses import dataclass

@dataclass
class KellyConfig:
    """
    Configuración del sistema Kelly fraccional.
    - kelly_fraction: porcentaje del Kelly completo (0.5 = medio Kelly)
    - max_risk_pct: riesgo máximo permitido por apuesta (% de la banca)
    - min_stake: apuesta mínima absoluta en USD
    """
    kelly_fraction: float = 0.5
    max_risk_pct: float = 0.05
    min_stake: float = 1.0


def kelly_stake(odds: float, prob: float, bankroll: float, cfg: KellyConfig) -> float:
    """
    Calcula el tamaño de la apuesta óptima según la estrategia de Kelly fraccional.

    Parámetros:
    - odds: cuota decimal del pick.
    - prob: probabilidad de éxito (entre 0 y 1).
    - bankroll: saldo total disponible.
    - cfg: configuración de Kelly (KellyConfig).

    Retorna:
    - Tamaño de la apuesta recomendada en USD (float).
    """

    # Validaciones seguras
    if bankroll <= 0 or odds <= 1 or not (0 <= prob <= 1):
        return 0.0

    # Kelly puro (sin límites)
    b = odds - 1.0
    if b <= 0:
        return 0.0

    edge = (b * prob - (1 - prob)) / b
    edge = max(edge, 0.0)  # no apostar si el valor es negativo

    # Apuesta ideal sin límites
    stake_raw = bankroll * edge * cfg.kelly_fraction

    # Limitar riesgo máximo por apuesta
    stake_cap = bankroll * cfg.max_risk_pct
    stake = min(stake_raw, stake_cap)

    # Aplicar mínimo absoluto si hay ventaja (edge > 0)
    if edge > 0 and stake < cfg.min_stake:
        stake = min(cfg.min_stake, stake_cap)

    # Redondear a 2 decimales
    return round(stake, 2)
