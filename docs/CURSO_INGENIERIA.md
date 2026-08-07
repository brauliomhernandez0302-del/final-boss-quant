# Curso de ingeniería del proyecto

Índice de referencia de **todo lo que hay que saber** para construir un sistema
de predicción deportiva con detección de valor, organizado por materia.

**Qué es esto**: un mapa de qué estudiar, en qué orden, con el recurso concreto
y —lo que lo hace útil— **el fallo real que este proyecto tuvo por no saberlo**.
Cada materia lleva su lista de errores cometidos, con fecha.

**Qué NO es**: un libro de texto. No reproduce el material; lo apunta.

**Cómo se usa**: cuando haya que decidir algo de una materia, se lee su sección
antes de escribir código, no después.

---

## Orden de estudio recomendado

Las materias no son independientes. Este orden minimiza el retrabajo:

```
1. Microestructura de mercado   ← sin esto, todo lo demás apunta al lugar equivocado
2. Probabilidad aplicada        ← el lenguaje mínimo
3. Ingeniería de datos          ← lo que hace que las mediciones signifiquen algo
4. Evaluación y medición        ← la balanza, antes que cualquier modelo
5. Modelado predictivo          ← recién acá
6. Gestión de banca
7. Dominio (sabermetría)        ← se puede comprar; las anteriores no
8. Producto y negocio
```

**La lección más cara del proyecto anterior fue hacer esto en el orden 5→7→1.**
Construyó el modelo primero, el dominio después, y la microestructura nunca.

---

## 1. Microestructura de mercados de apuestas

**Por qué va primera**: determina si el problema que elegiste tiene solución.
Pelear el moneyline de MLB contra Pinnacle y pelear un prop de pitcher en una
casa blanda son dificultades separadas por un orden de magnitud, y la
diferencia no está en el modelo.

### Qué hay que saber

- **Cómo se forma un precio.** Los libros **copian las líneas de los
  originadores** en vez de manejar los partidos ellos mismos, y esperan a que
  los apostadores sharp les formen la línea. De ahí sale el rezago como
  propiedad estructural, no como accidente.
- **Vig y hold.** Mercados principales 4-5%, **props 6-10%**. El peaje define
  cuánto edge hace falta antes de que quede algo.
- **Devig.** Cuatro métodos (multiplicativo, potencia, Shin, aditivo). Cuál
  usar depende del mercado. Desvigorizar un libro blando da una estimación
  mucho peor que desvigorizar uno sharp y líquido.
- **Market width.** La dispersión entre libros sharp. **Ancho chico = confianza
  alta; ancho grande = nadie sabe el precio y vos tampoco.** Es el filtro de
  calidad de cualquier discrepancia.
- **CLV.** Ganarle al precio de cierre es *la* medida de habilidad. También es
  lo que dispara que te limiten.
- **Límites.** Una vez marcado: un par de cientos en mercados principales,
  **$50 en props**. Pinnacle, Kalshi y Polymarket no limitan.
- **La ventaja del tomador.** El creador de mercado está **obligado** a cotizar
  los dos lados; el tomador elige cuándo entrar. Esa asimetría es el activo.

### Recursos

- Ed Miller & Matthew Davidow — *The Logic of Sports Betting* — **el primero
  que hay que leer, sin discusión**
- Joseph Buchdahl — *Squares and Sharps, Suckers and Sharks*
- Unabated — material educativo sobre devig y línea sharp
- Hubáček & Šír — https://arxiv.org/abs/2010.12508

### Errores cometidos por no saberlo

- **2026-08-04**: se midió el modelo contra el moneyline, el mercado más
  eficiente que existe, durante meses. La pregunta estaba mal formulada desde
  el principio.
- **CLV medido contra precio crudo** en vez de desvigorizado: regalaba el vig
  entero como habilidad. Media +1.514% pasó a **−0.497%** al corregirlo. *El
  único indicador de habilidad positivo del proyecto era un artefacto del vig.*

---

## 2. Probabilidad y estadística aplicada

### Qué hay que saber

- **Cuota ↔ probabilidad ↔ EV.** `EV = p·(cuota−1) − (1−p)`. Trivial y aun así
  es donde se cuelan los errores de emparejamiento.
- **Varianza y tamaño de muestra.** A n=500 apuestas con cuota ~2, el error
  estándar del ROI ronda **4.5%**. Un +2% no distingue nada.
- **Cuántas apuestas para probar habilidad**: por ROI, **más de 1.000** con
  p<0.001. Por CLV, **65**. Quince veces menos.
- **Errores agrupados.** Observaciones que se repiten por entidad (mismo
  equipo, mismo pitcher) no son independientes. Los errores iid sobre ellas
  **inflan los t entre 5x y 21x**.
- **Bootstrap por clúster**, no por fila.
- **Regresión a la media.** Para un récord de MLB: sumarle **~67 juegos de
  .500** para estimar talento verdadero.

### Recursos

- Buchdahl sobre CLV y significancia — https://www.pinnacleoddsdropper.com/blog/closing-line-value--clv-demystified-by-expert-joseph-buchdahl
- Gelman & Hill — *Data Analysis Using Regression and Multilevel Models*

### Errores cometidos

- **Tres falsos positivos** en la auditoría del pipeline (QS% con t=+4.79,
  "suerte" con t=−21, la ventaja del compuesto TTE) que **desaparecieron al
  agrupar los errores**.
- **2026-08-05**: el portón dejó pasar una señal de **ruido puro** porque
  evaluaba ROI sobre ~580 apuestas. De ahí salió `MIN_APUESTAS=500`.

---

## 3. Ingeniería de datos

### Qué hay que saber

- **Point-in-time correctness.** El concepto central de un *feature store*: el
  histórico completo vive con valores correctos al momento. Es lo que llamamos
  "contrato de tiempo".
- **Append-only / inmutabilidad.** Un precio observado es un hecho histórico.
  Sobreescribirlo destruye la única serie que no se puede volver a comprar.
- **Las reglas las impone el mecanismo.** Triggers, `NOT NULL`, tipos,
  constructores que rechazan. Nunca la disciplina del llamador.
- **Idempotencia y procedencia.** Cada fila sabe de dónde vino y cuándo se supo.
  Una corrección **agrega**, no pisa.
- **Cortes estrictos.** `<`, nunca `<=`. Y comparar instantes, no cadenas.
- **Separar hecho de opinión.** Nunca en la misma tabla.
- **Stack para un equipo chico en 2026**: DuckDB + Parquet + Python, con dbt
  cuando las transformaciones pasen de unas pocas. Nada de Spark/Kafka.

### Recursos

- Databricks — https://www.databricks.com/blog/what-feature-store-complete-guide-ml-feature-engineering
- KDnuggets — Python, Parquet y DuckDB
- Datacoves — dbt + DuckDB

### Errores cometidos

- **CHRON-001**: hechos y predicciones en la misma tabla. Una corrida de
  backtest sobrescribió **563 filas** de predicciones en vivo. Cero
  recuperables.
- **Leak V4**: el día del juego salía del timestamp UTC truncado en vez del
  `officialDate`. **22-24% de los juegos** caían un día adelante.
- **Doce capturas de precio por día, una guardada.** El docstring lo describía
  como feature.
- **Una caché de ofensa congelada a mitad de temporada**, usada en silencio por
  el **59%** de los juegos del backtest insignia.
- **2026-08-05**: `pair_before` devolvía precios EN VIVO mezclados con
  pre-juego. **31.5%** de lo capturado era post-inicio, con moneylines de 150.0.

---

## 4. Evaluación y medición

**Va antes del modelado. Siempre.**

### Qué hay que saber

- **Reglas de puntuación propias.** Brier y log-loss para binarios; **CRPS**
  para distribuciones. "Estrictamente propia" significa que **no podés mejorar
  el puntaje mintiendo sobre tu creencia verdadera**. El MAE sí se puede gamear.
- **Descomposición de Murphy**: `Brier = incertidumbre − resolución +
  fiabilidad`. Distingue "mal calibrado" de "no discrimina", que se arreglan de
  formas opuestas.
- **Calibración vs precisión.** Para apostar, **calibración**. Pero ojo con el
  paper de Walsh & Joshi: sus números fueron **corregidos** y en el caso medio
  ambos pierden (−9.77% vs −26.78%). *Calibrás y perdés menos*, no *calibrás y
  ganás*.
- **El nulo correcto es el mercado**, no el azar ni tu versión anterior.
- **Regresión conjunta.** `y ~ logit(mercado) + logit(candidato)`. Si el
  coeficiente del candidato no es positivo, no aporta nada dado el precio.
- **Fuera de muestra siempre**, con pliegues **temporales**, y estandarizando
  con estadísticos **del pliegue de entrenamiento únicamente**.
- **El ROI por umbral de edge es gate obligatorio.** Brier y ROI pueden moverse
  en direcciones opuestas.
- **Control positivo obligatorio.** Un instrumento que no detecta una señal
  plantada tampoco detectaría una real.

### Recursos

- Gneiting & Raftery — https://arxiv.org/pdf/1709.04743
- Walsh & Joshi — https://arxiv.org/abs/2303.06021 (+ el corrigendum)

### Errores cometidos

- **La balanza llegó décima en vez de tercera.** Siete baselines en tres meses,
  todos caídos por defectos de medición.
- `defense` salía significativo **en muestra** (P(b>0)=99.0%) y resultó el
  **peor de siete** fuera de muestra.
- El instrumento de cobertura promediaba sobre una ventana que cruzaba
  arreglos: un campo roto **ayer** aparecería al 90% y pasaría limpio.

---

## 5. Modelado predictivo

### Qué hay que saber

- **Regresión distribucional.** Un prop pregunta `P(X > línea)`, no `μ`.
  **NGBoost** devuelve la distribución completa usando el gradiente natural.
- **Modelos de conteo.** Poisson, **binomial negativa** (sobredispersa),
  inflada en cero. Los ponches y las carreras limpias son binomial negativa.
- **Jerárquico bayesiano / partial pooling.** Cada jugador se encoge hacia la
  media poblacional **en proporción inversa a su muestra**, automáticamente.
  Sustituye a las constantes de encogimiento a mano.
- **Calibración posterior.** Platt se sobreajusta con conjuntos chicos; **con
  n≥1000 la isotónica siempre iguala o supera**.
- **Decorrelación del mercado.** `Loss = error + λ·correlación(residuos,
  precio)`. **Un modelo preciso es no rentable si está correlacionado con el de
  la casa.**
- **Compresión.** Cada capa entre el dato y la apuesta destruye información.
  xwOBA → λ → 7 multiplicadores → Monte Carlo → P(gana) son cinco.

### Recursos

- NGBoost — https://arxiv.org/abs/1910.03225
- PyMC partial pooling — https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html
- Niculescu-Mizil & Caruana — https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
- Hubáček, Šourek & Železný — https://ida.fel.cvut.cz/papers/hubacek2019exploiting.html

### Errores cometidos

- **Nueve motores construidos, ninguno pasó nunca por un portón.** Al pasarlos
  en 2026-08-04: las 14 celdas negativas.
- `l0` (TTE de Statcast) tenía coeficiente **negativo** condicionado al precio:
  explicaba el **42.5%** de la opinión del mercado. Un mal duplicado.
- **Encogimiento por constantes a mano** (`K_BARREL=120`,
  `PRIOR_PA_EQUIVALENT=1000`) en vez de jerárquico.

---

## 6. Gestión de banca

### Qué hay que saber

- **Kelly**: `f = (p·b − q)/b`. Maximiza crecimiento logarítmico.
- **Kelly con probabilidad estimada sobre-apuesta sistemáticamente.** La
  corrección: encoger en proporción a la **incertidumbre de la estimación**.
- **Kelly bayesiano**: con creencias beta, **40-60% menos drawdown máximo
  conservando 85-95% del crecimiento** frente a Kelly completo y a fraccional
  ad-hoc.
- **Teoría de portafolio** para apuestas simultáneas y correlacionadas. Es uno
  de los tres ingredientes del método de Hubáček.
- **Un piso de stake puede inflar un edge marginal.** Caso real: 0.28% de Kelly
  convertido en 1.00% por el piso — 3.6x.

### Recursos

- Baker & McHale — https://pubsonline.informs.org/doi/abs/10.1287/deca.2013.0271
- Kelly bayesiano — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6195358

### Errores cometidos

- `KELLY_FRACTION=0.25` fijo: el "fraccional ad-hoc" que la literatura señala
  como inferior.
- Por PURP-1, el EV inflado agrandaba las apuestas vía Kelly: **508 de 822
  unidades arriesgadas —el 61.8% del capital— en picks con edge fabricado**,
  con stake medio 13.04u contra 3.82u del resto.

---

## 7. Dominio — sabermetría

**Es la única materia que se puede comprar.** Las seis anteriores, no.

### Qué hay que saber

- **Métricas esperadas**: xwOBA, xERA, xFIP, SIERA, barrel%, exit velocity,
  launch angle. Separan resultado de proceso.
- **Modelos de *stuff***: velocidad, movimiento, spin, tunneling. Miden talento
  del brazo independiente del resultado.
- **Factores de parque**, y que interactúan con clima.
- **Estimación de talento verdadero**: récord < Pythagorean < **BaseRuns**. El
  diferencial de carreras regresiona menos que las victorias.
- **Indicadores extra-deportivos**, con magnitud: **el árbitro de home vale 1-2
  ponches en un prop de pitcher y ~0.5 carreras en el total.** Clima,
  temperatura, viento, descanso, viaje.
- **Que ningún sistema de proyección domina**: ZiPS mejor para wOBA de
  bateadores, Steamer para wOBA permitido, y **el compuesto le gana a todos por
  separado**.

### Recursos

- FanGraphs Sabermetrics Library — el estándar, gratis
- `pybaseball` — https://github.com/jldbc/pybaseball — Statcast, BR, FanGraphs
  y **el registro Chadwick para traducir IDs entre fuentes**
- Baseball Savant, Retrosheet
- THE BAT X ($99/mes) — proyecciones premium en CSV

### Errores cometidos

- La misma constante de barrel% arreglada en el motor vivo **y no en la copia
  PIT**, que era una implementación duplicada.
- `batted_ball_count` inflado **1.905x** por contar fouls no terminales — en
  **dos implementaciones independientes**.

---

## 8. Producto y negocio

### Qué hay que saber

- **Vender picks es legal** en EE.UU. y no requiere licencia, pero debe llevar
  descargo de entretenimiento/consultoría. **Las tarifas por rendimiento son
  riesgo regulatorio**: tarifa plana o suscripción.
- **La FTC en 2026** mira las suscripciones: revelar términos **antes** de pedir
  facturación, y no dificultar la cancelación.
- **Techo de capacidad.** *Una vez publicados los picks, los mercados se mueven
  o los bajan en minutos.* Un pick bueno se autodestruye al publicarse; una
  herramienta no.
- **La aritmética del suscriptor**: con **$50 de límite en props** y 5% de edge
  son $2.50 por apuesta. Una membresía de $20 necesita **8 apuestas mensuales
  sólo para empatarse**.
- **Verificación de terceros** (Juice Reel, Pikkit, betstamp) como sello. No lo
  decís vos.
- **El foso de los sindicatos no es el modelo: son los datos propios y la
  ejecución.** Starlizard tiene ~100 analistas *generando* datos. Y su problema
  más caro —meter el dinero sin que lo limiten— **no existe si vendés
  información**.

### Recursos

- FTC, prácticas de suscripción 2026
- Estructura de OddsJam y Unabated como productos

---

## Cómo se mantiene este documento

Cada vez que se comete un error que una de estas materias habría evitado, **se
agrega a la lista de errores de esa materia, con fecha**. La lista es el activo:
un curso genérico está en cualquier lado; el registro de *nuestros* fallos, no.
