# Fase 2B — B3: MATH-003 residual re-medido sobre el baseline des-leakeado

**Fecha**: 2026-07-19. Comando: `python3 scripts/math003_home_win_residual.py` (mismo script
repetible de 5d, sin cambios), corrido sobre `data/predictions_history.db` post-B2.

## Resultado

| Temporada | Home win rate real | p_home_raw CON término (actual) | Residual CON | p_home_raw SIN término (reestimado) | Residual SIN |
|---|---|---|---|---|---|
| 2024 (n=2429) | 52.16% | 52.03% | +0.13pp | 50.40% | +1.76pp |
| 2025 (n=2430) | 54.28% | 52.76% | +1.52pp | 51.12% | +3.16pp |

## Comparación contra la medición de 5d (pre-Fase 2B, con el leak V4 todavía activo)

| | 2024 residual CON | 2024 residual SIN | 2025 residual CON | 2025 residual SIN |
|---|---|---|---|---|
| Antes (5d, con leak) | +0.08pp | +1.70pp | +1.47pp | +3.10pp |
| Después (B3, sin leak) | +0.13pp | +1.76pp | +1.52pp | +3.16pp |
| Δ | +0.05pp | +0.06pp | +0.05pp | +0.06pp |

## Veredicto — SIN implementar, decisión sigue abierta

El leak V4 apenas movió este residual específico (±0.05-0.06pp en las 4 celdas) — el mismo
patrón cualitativo de 5d se sostiene con datos des-leakeados: **2024 queda casi cerrado**
(+0.13pp con el término activo), **2025 retiene un residual real de ~1.5pp sin corregir**. El
leak V4 afectaba la precisión general del modelo (Brier/accuracy, un efecto amplio y disperso a
través del entrenamiento walk-forward de team-bias), pero no estaba concentrado
específicamente en el sesgo de probabilidad de victoria de local que `_UNIFORM_HOME_MULT`
corrige — por eso el residual de home-win apenas se movió pese a que el Brier/accuracy general
sí se movieron sustancialmente.

**No se cambia `_UNIFORM_HOME_MULT`**, tal como exige esta fase. La decisión sigue exactamente
donde 5d la dejó — mantenerlo en 0.028 (buen compromiso para 2024, insuficiente para 2025) o
investigar por qué 2025 tiene un gap real casi el doble que 2024 antes de tocar la constante —
ahora simplemente con la confirmación de que esa decisión no depende del leak V4 que se acaba
de arreglar.
