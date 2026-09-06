# Veredictos de features — v2, sobre el almacén propio

**Referencia de esta versión**: almacén propio (`market.db` + `results.db`),
**n = 6.106**, **Brier del nulo = 0,242124**, precios posteriores al primer
lanzamiento excluidos. Ver `docs/REFERENCIA_MERCADO_2026-09.md`.

La v1 **no se reemplaza ni se corrige**: se midió contra otro nulo y sigue
siendo cierta sobre su propia referencia. Las dos conviven en el registro de
cada feature (`fbq/features/market_shape.py`, campo `veredictos`), cada una con
su fecha, su almacén, su referencia y el commit que la produjo.

## Los dos veredictos, lado a lado

| feature | v1 (2026-08-04, legado, n=5.429, Brier 0,241560, `9b7c33a`) | v2 (2026-09-06, propio, n=6.106, Brier 0,242124) |
|---|---|---|
| `desacuerdo_cons_pin` | **NO CRUZA** — signo positivo y **estable**: +0,0387 / +0,0755 / +0,0554 | **NO CRUZA** — signo **inestable**: +0,0298 / −0,0002 / −0,0120 |
| `prima_mejor_precio` | **NO CRUZA** — signo negativo y **estable**: −0,0594 / −0,0547 / −0,0432 | **NO CRUZA** — signo **inestable**: −0,0722 / +0,0060 / +0,0267 |
| `profundidad_mercado` | **NO CRUZA** — signo cambiante: −0,0662 / +0,0267 / +0,0210 | **NO MEDIBLE** — el dato no existe en el almacén propio |

Las tres siguen sin cruzar. Lo que cambió es **por qué**, y eso importa.

## El hallazgo: la "estabilidad de signo" la sostenían los precios en vivo

La v1 destacaba de `desacuerdo` y `prima` que su coeficiente era estable en las
tres temporadas — *"estabilidad de signo que ninguno de los nueve motores del
sistema anterior tuvo"*. Era el único rasgo positivo que el proyecto le había
encontrado a una feature propia.

**No sobrevive.** La causa está aislada en tres pasos, cada uno cambiando una
sola cosa:

### `desacuerdo_cons_pin`

| | 2024 | 2025 | 2026 | n de 2026 |
|---|---|---|---|---|
| **A** · camino exacto de la v1, con los datos de hoy | **+0,0387** | **+0,0755** | **+0,0554** | 775 |
| **B** · igual, pero excluyendo los juegos con precio en vivo | +0,0201 | **−0,0164** | **−0,0120** | 751 |
| **C** · v2 completa (feature y nulo del almacén propio) | +0,0298 | −0,0002 | −0,0120 | 1.532 |

### `prima_mejor_precio`

| | 2024 | 2025 | 2026 | n de 2026 |
|---|---|---|---|---|
| **A** · camino exacto de la v1, con los datos de hoy | **−0,0594** | **−0,0547** | **−0,0432** | 775 |
| **B** · igual, pero excluyendo los juegos con precio en vivo | −0,0638 | **+0,0304** | **+0,0267** | 751 |
| **C** · v2 completa | −0,0722 | +0,0060 | +0,0267 | 1.557 |

**Lectura**:

1. **La fila A reproduce la v1 al último decimal.** La v1 no era un error de
   medición ni cambió por el paso del tiempo: es exactamente reproducible.
2. **El salto ocurre entre A y B**, y entre A y B cambia **una sola cosa**:
   excluir 129 juegos (2,4%) cuyo precio se observó después del primer
   lanzamiento. Eso basta para dar vuelta el signo de 2025 y 2026 en las dos
   features.
3. **La fila C** —cambiar además el nulo y ampliar 2026 de 775 a ~1.550 juegos—
   mueve poco respecto de B. El cambio de almacén no es el responsable.

**Comprobación adicional**: los valores de la feature son **idénticos bit a
bit** en los 5.300 juegos que las dos implementaciones cubren. La vieja leía
`historical_odds`; la nueva lee `market.db`. Cero diferencias. Lo único que
cambia es la cobertura: 129 juegos menos, y son exactamente los de precio en
vivo.

**Conclusión**: la única propiedad atractiva que estas dos features tenían era
un artefacto de 129 cotizaciones que ya habían visto parte del partido. Al
medirlas sólo con precios que un apostador podía tomar de verdad, la señal no
tiene un signo consistente.

## `profundidad_mercado`: NO MEDIBLE no es NO CRUZA

Su dato es `n_bookmakers` —cuántas casas cotizaban, entre 2 y 32— y sólo existía
en `historical_odds`. De ese período la importación conservó **tres precios por
juego** (Pinnacle, el consenso y el mejor), no el board completo, así que contar
libros en el almacén propio mediría **qué guardó el sistema anterior**, no la
profundidad del mercado. Inventar el número no es una opción.

La implementación nueva cuenta casas reales y se restringe a los juegos que
capturamos nosotros: hoy son **25**, y el portón lo reporta como **NO MEDIBLE**
en vez de como un veredicto sobre la señal. Se vuelve medible sola a medida que
la captura propia acumule temporadas.

El veredicto vigente **sobre la señal** sigue siendo el **v1**, con su
referencia original.

## Reproducir

```bash
python3 -m fbq.features                    # v2: nulo y features del almacén propio
python3 -m fbq.features --almacen legado   # cambia SÓLO el nulo — NO reproduce la v1
```

⚠️ `--almacen legado` cambia el nulo pero **no** la fuente de las features, que
desde el 2026-09-06 leen del almacén propio. Para reproducir la v1 hay que
correr el código de su commit, `9b7c33a`. El CLI lo advierte al usarlo.

## Migración completada

`fbq/features/` ya no lee `predictions_history.db`:

| archivo | antes | ahora |
|---|---|---|
| `features/gate.py` | `load_frame()` (legado) por defecto | `load_frame_propio()` por defecto; `almacen="legado"` sigue disponible |
| `features/market_shape.py` | `sqlite3.connect(predictions_history.db)` | consulta `market.db`, última cotización **pre-juego** por libro |

Con esto, `core/`, `sources/`, `market/`, `results/`, `evaluator/` y `features/`
—los pasos 0 a 6 del build— corren **enteramente sobre almacenes propios**.
