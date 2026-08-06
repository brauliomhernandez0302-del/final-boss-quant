# Investigación — 2026-08-05

Constancia de la investigación previa a definir las secciones del build. Todo lo
de acá está verificado contra fuentes públicas, con enlace. Lo que es afirmación
de un vendedor está marcado como tal.

---

## 1. La tesis que replantea el problema

**Hubáček & Šír, *Beating the market with a bad predictive model*** (arXiv
2010.12508), y su antecedente **Hubáček, Šourek & Železný, *Exploiting
sports-betting market using machine learning*** (International Journal of
Forecasting, 2019).

**Tesis, textual: es posible obtener ganancias sistemáticas con un modelo de
predicción completamente inferior.** El mecanismo es alterar el objetivo de
entrenamiento para **decorrelacionar el modelo del mercado**.

### Por qué funciona

El apostador gana cuando su estimación `T` y el precio `M` están desalineados
respecto del valor real `R` de dos formas concretas:

1. el mercado subvalúa (`m < r`) **y** el apostador estima más alto (`t > m`), o
2. el mercado sobrevalúa (`m > r`) **y** el apostador estima más bajo (`t < m`)

En esas dos situaciones **el tomador no es penalizado por sus propios errores de
estimación**, y ocurren más veces que las contrarias. Con distribuciones
uniformes el paper reporta una **ventaja de 2:1**.

La asimetría estructural: **el creador de mercado está obligado a cotizar los
dos lados; el tomador tiene la libertad de elegir sólo las oportunidades que le
convienen.** Esa libertad es el activo, no la precisión.

### El objetivo de entrenamiento

En vez de minimizar el error de predicción, se maximiza la independencia entre
el residuo del modelo `(T − R)` y el del mercado `(M − R)`:

```
Loss = ErrorDePrediccion + λ · Correlacion(ResiduosDelModelo, PreciosDeMercado)
```

Y la frase que aplica directo a lo que medimos: **un modelo preciso sigue siendo
no rentable mientras esté correlacionado con el modelo de la casa.**

### Cómo se conecta con lo medido acá el 2026-08-04

- Brier del modelo 0.24620 contra 0.24052 del mercado → **inferior**
- `l0` explicaba el **42.5%** de la opinión del mercado, con coeficiente
  condicionado **negativo**

O sea: era un mal duplicado del precio. **El problema no era ser peor; era ser
peor Y correlacionado.** No hay dónde ganar si te equivocás en los mismos sitios
que el mercado y además más.

Resultado empírico del paper de 2019: beneficios acumulados positivos y
sistemáticos sobre NBA 2007-2014, con tres ingredientes — decorrelación, CNN
sobre estadísticas de jugador, y teoría de portafolio para distribuir apuestas.

---

## 2. Calibración por encima de precisión — con una corrección importante

**Walsh & Joshi (Universidad de Bath), *Machine learning for sports betting:
should model selection be based on accuracy or calibration?*** (Machine Learning
with Applications, 2024).

Entrenan sobre varias temporadas de NBA y apuestan sobre una temporada con odds
publicadas.

**Conclusión: elegir el modelo por calibración da mejor retorno que elegirlo por
precisión.**

**PERO los números originales fueron corregidos**, y la corrección cambia la
lectura:

| | original | **corregido** |
|---|---|---|
| ROI medio (calibración vs precisión) | +34.69% vs −35.17% | **−9.77% vs −26.78%** |
| ROI mejor caso | +36.93% vs +5.56% | **+23.13% vs +10.9%** |

Calibración sigue ganando, **pero en el caso medio ambos PIERDEN plata.** La
lección honesta no es "calibrá y ganás" — es "calibrá y perdés menos". Elegir por
precisión es peor; elegir por calibración no alcanza por sí solo.

Razón conceptual: la precisión sirve para acertar quién gana; **el problema de
apostar es estimar la probabilidad verdadera** para detectar el precio mal puesto.

---

## 3. Las técnicas de ML que corresponden

### Regresión distribucional — NGBoost

Devuelve **la distribución completa**, no un punto. Trata los parámetros de la
distribución condicional como objetivos del boosting y usa el **gradiente
natural**, que respeta la geometría del espacio de probabilidades; el gradiente
ordinario es inestable para aprender distribuciones multiparamétricas.

Funciona con cualquier learner base, cualquier familia con parámetros continuos
y cualquier regla de puntuación.

**Por qué importa acá**: un prop pregunta `P(X > línea)`. Una proyección de DFS
—incluida THE BAT X— vende **μ**. La dispersión alrededor de μ no la vende
nadie, y es la que convierte una proyección en un precio.

### La métrica — CRPS, no MAE

**CRPS** (Matheson & Winkler 1976) es una regla de puntuación **estrictamente
propia** que evalúa la distribución completa y por lo tanto **mide calibración y
agudeza simultáneamente** (Gneiting & Raftery 2007).

"Estrictamente propia" significa que **no se puede mejorar el puntaje mintiendo
sobre la creencia verdadera**. Se expresa en la unidad de la observación y se
reduce al error absoluto si el pronóstico es determinista.

### Muestras chicas — jerárquico bayesiano

*Partial pooling* es la implementación principiada de la regresión a la media:
cada jugador se encoge hacia la media poblacional **en proporción inversa a su
tamaño de muestra**, automáticamente. Cuando la varianza muestral es chica
respecto de la varianza entre grupos, el factor de encogimiento tiende a 0 y el
grupo conserva su estimación cruda.

Sustituye a las constantes ajustadas a mano (`K_BARREL=120`,
`PRIOR_PA_EQUIVALENT=1000`) del sistema anterior.

### Calibración — isotónica, con condición

- **Platt se sobreajusta con conjuntos de calibración chicos** y no maneja bien
  la varianza alta.
- **Con 1.000 puntos o más, la isotónica siempre iguala o supera a Platt**
  (Niculescu-Mizil & Caruana).

### Sizing — Kelly bayesiano

Kelly con probabilidad estimada en vez de verdadera **sobre-apuesta
sistemáticamente**; el rendimiento fuera de muestra es peor que dentro. La
corrección es **encoger la apuesta en proporción a la incertidumbre de la
estimación**.

Un trabajo con creencias beta reporta **40-60% menos drawdown máximo
conservando 85-95% de la tasa de crecimiento** frente a Kelly completo y a
fraccional ad-hoc. NGBoost entrega la incertidumbre, así que el encogimiento
sale del modelo y no de una constante.

---

## 4. La realidad operativa — límites

Esto condiciona el producto más que cualquier decisión técnica.

- En casas legales **los límites llegan rápido**. Una vez marcado, es raro poder
  apostar más de un par de cientos en mercados principales — y **$50 en props y
  derivados**.
- Lo que te marca: **que tus apuestas le ganen sistemáticamente a la línea de
  cierre.** Es decir, el CLV es a la vez la prueba de habilidad y el disparador
  de la limitación.
- Apostar líneas de apertura es una bandera roja grande.
- **Pinnacle, Kalshi y Polymarket no limitan a los ganadores.**
- La marca se comparte entre casas: limitado en una, limitado preventivamente en
  otras.

### La aritmética que esto impone

Si el suscriptor sólo puede poner **$50 en un prop**, con un edge del 5% gana
**$2.50 por apuesta**. Una membresía de $20/mes necesita **8 apuestas mensuales
sólo para empatar la suscripción**.

No invalida el negocio, pero define su forma: el volumen de picks y el tamaño
del edge tienen que ser suficientes para que la suscripción se pague sola, y eso
hay que calcularlo antes de fijar precio.

---

## 5. La capacidad — el límite que nadie menciona

De la descripción del propio programa de picks de Derek Carty:

> Una vez que se publican los picks, los mercados se mueven dramáticamente o los
> bajan en cuestión de minutos.

**Un pick bueno se autodestruye al publicarse.** Cuantos más suscriptores, más
rápido muere la línea. El servicio de picks tiene un **techo de capacidad** que
no lo pone el modelo sino el mercado.

Es, con evidencia, por qué los que escalan venden herramientas y no picks: una
pantalla no se satura porque mil usuarios apuestan cosas distintas.

---

## 6. Lo legal (EE.UU.)

- **Vender picks es legal** y el mercado está en gran medida no regulado; no
  requiere licencia. No debe facilitar el juego en sí.
- **Descargos obligatorios**: entretenimiento y consultoría, no consejo de juego
  garantizado.
- **Las tarifas por rendimiento son riesgo regulatorio.** Tarifa plana o
  suscripción.
- **Suscripciones, foco de la FTC en 2026**: hay que revelar todos los términos
  materiales *antes* de pedir datos de facturación, y no dificultar la
  cancelación.
- Si la oferta cruza estados, hay que considerar las reglas de cada uno.
- LLC para protección de responsabilidad.

---

## 7. Datos y proveedores

| | The Odds API | OpticOdds |
|---|---|---|
| Precio | $30-249/mes | empresarial, a medida |
| Casas | ~40 medidas (us, us2, eu) | 200+ |
| Latencia | **sólo polling REST, sin streaming** | **streaming sub-800ms** |
| Histórico | sí, en todos los planes | — |
| Props | sí, selectivo | sí |

**Consecuencia**: con The Odds API el producto de **cazar rezago de precio no es
viable** — se compite contra streaming sub-segundo. El producto de **props
publicados la noche anterior sí lo es**, porque el edge no viene de la velocidad.

Cuota estimada para props de MLB: ~15 juegos × 5 mercados × 12 barridas ≈
**27.000 créditos/mes**, dentro del plan de $59.

### Proyecciones

**THE BAT X** (Derek Carty): $19.99 por 3 días, $99.99/mes, $499.99/año. CSV
descargable de bateadores y pitchers. Modelo de *stuff* con Statcast desde 2026
(velocidad, movimiento, spin, tunneling), más parque, clima, árbitros, cátcher,
bullpen, conteo de lanzamientos, platoon, defensa y posición en alineación.

**Advertencia sobre su precisión**, medida por FanGraphs y no por el vendedor:
ZiPS fue mejor para wOBA de bateadores, Steamer para wOBA permitido por
pitchers, y **el compuesto —promedio de sistemas— le ganó a todos por
separado**. THE BAT X es mucho más preciso con veteranos que con novatos.

ZiPS y Steamer son **gratuitos** en FanGraphs. Un compuesto de gratuitos puede
igualar a un sistema pago.

---

## 8. Ingeniería

- Lo que llamamos "contrato de tiempo" se llama **point-in-time correctness** y
  es el concepto central de un *feature store*.
- Stack para equipo de una persona en 2026: **DuckDB + Parquet + Python**, con
  **dbt** cuando las transformaciones pasen de unas pocas. Nada de Spark/Kafka.
- Errores sistemáticos en las guías populares de "cómo construir un modelo de
  apuestas": corte train/test **aleatorio** sobre datos temporales; métricas
  MAE/RMSE/R² en vez de calibración y CLV; **ninguna mención del vig**; ninguna
  mención de point-in-time; y **nada sobre la distribución**, que es el problema.

---

## Fuentes

- Hubáček & Šír — *Beating the market with a bad predictive model* — https://arxiv.org/abs/2010.12508
- Hubáček, Šourek & Železný — *Exploiting sports-betting market using machine learning* — https://ida.fel.cvut.cz/papers/hubacek2019exploiting.html
- Walsh & Joshi — *accuracy or calibration?* — https://arxiv.org/abs/2303.06021 · corrigendum: https://www.sciencedirect.com/science/article/pii/S2666827025000106
- NGBoost — https://arxiv.org/abs/1910.03225 · https://stanfordmlgroup.github.io/projects/ngboost/
- Gneiting & Raftery, reglas de puntuación propias — https://arxiv.org/pdf/1709.04743
- PyMC, partial pooling jerárquico — https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html
- Niculescu-Mizil & Caruana, calibración — https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
- Kelly bayesiano con incertidumbre — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6195358
- Baker & McHale, *Optimal Betting Under Parameter Uncertainty* — https://pubsonline.informs.org/doi/abs/10.1287/deca.2013.0271
- Límites de casas — https://howgamblingworks.substack.com/p/the-truth-about-limits
- The Odds API — https://the-odds-api.com/#get-access
- OpticOdds — https://opticodds.com/sports-betting-api
- THE BAT X — https://rotogrinders.com/marketplace/derek-carty-s-the-bat-projection-system-300
- FanGraphs, comparación de sistemas — https://fantasy.fangraphs.com/2026-projection-showdown-the-bat-x-vs-steamer-era-forecasts/
- Carty, show de props en Covers — https://www.covers.com/mlb/the-bat-x-release-show
- Databricks, feature store — https://www.databricks.com/blog/what-feature-store-complete-guide-ml-feature-engineering
