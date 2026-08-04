"""Las constantes del TTE no pueden divergir entre el camino vivo y el PIT.

QUÉ PASÓ ANTES
==============
`offense/true_talent_engine.py` (vivo) y `advanced_pit_enrichment/
tte_pit_adapter.py` (backtest) tenían copias independientes de la misma
fórmula. Derivaron: la constante de barrel% se arregló en una copia y no en la
otra. Efecto medido y documentado en el docstring de `tte_formula.py`: los dos
caminos sólo correlacionaban **0.733** en λ real por equipo, con desacuerdos de
hasta ±0.4-0.5 carreras. Eso es MATH-002.

El 2026-07-12 se creó `tte_formula.py` como fuente única y se unificó la
MATEMÁTICA. Pero las CONSTANTES siguen duplicadas:

    K_XWOBA, K_BARREL, K_BB, K_K        definidas por separado en los dos, sin
                                        fuente compartida
    LG_BB_PCT, LG_K_PCT, LG_BARREL_PA   idem
    PRIOR_PA_EQUIVALENT                 `tte_formula` la exporta y el adaptador
                                        PIT la importa, pero el motor VIVO
                                        mantiene su propia `_PRIOR_PA_EQUIVALENT`

Hoy los ocho valores coinciden. El mecanismo que produjo la deriva sigue
íntegro: basta que alguien recalibre una y olvide la otra.

POR QUÉ UN TEST Y NO UN REFACTOR
================================
Mover las constantes a `tte_formula` y importarlas es lo correcto, pero toca el
motor de λ en vivo y por regla de la casa eso exige el gate de ROI por umbral.
Este test es la defensa de coste cero: no cambia ni un decimal, y convierte la
deriva silenciosa en un fallo de CI. Cuando el refactor se haga con su gate,
este test sigue valiendo — pasará por construcción.
"""
import pytest

from modules.baseball_module.advanced_pit_enrichment import tte_pit_adapter as pit
from modules.baseball_module.offense import tte_formula as compartido
from modules.baseball_module.offense import true_talent_engine as vivo

# (nombre legible, atributo en el motor vivo, atributo en el adaptador PIT)
PARES = [
    ("K_XWOBA",             "_K_XWOBA",             "K_XWOBA"),
    ("K_BARREL",            "_K_BARREL",            "K_BARREL"),
    ("K_BB",                "_K_BB",                "K_BB"),
    ("K_K",                 "_K_K",                 "K_K"),
    ("PRIOR_PA_EQUIVALENT", "_PRIOR_PA_EQUIVALENT", "PRIOR_PA_EQUIVALENT"),
    ("LG_BB_PCT",           "LG_BB_PCT",            "LG_BB_PCT"),
    ("LG_K_PCT",            "LG_K_PCT",             "LG_K_PCT"),
    ("LG_BARREL_PA",        "LG_BARREL_PA",         "LG_BARREL_PA"),
]


@pytest.mark.parametrize("nombre,attr_vivo,attr_pit", PARES)
def test_la_constante_coincide_en_los_dos_caminos(nombre, attr_vivo, attr_pit):
    v = getattr(vivo, attr_vivo)
    p = getattr(pit, attr_pit)
    assert v == p, (
        f"{nombre} derivó: vivo={v}, PIT={p}. Es exactamente MATH-002 — dos "
        f"copias de la misma constante, una recalibrada y la otra no. Al "
        f"corregir una hay que corregir las dos, o mover la constante a "
        f"modules/baseball_module/offense/tte_formula.py e importarla en ambos."
    )


def test_prior_pa_equivalent_ya_tiene_fuente_unica():
    """La única de las ocho que ya está en `tte_formula`. El adaptador PIT la
    importa de ahí; el motor vivo aún guarda su copia. Mientras esa copia
    exista, este test es lo que impide que se separen."""
    assert compartido.PRIOR_PA_EQUIVALENT == vivo._PRIOR_PA_EQUIVALENT
    assert compartido.PRIOR_PA_EQUIVALENT == pit.PRIOR_PA_EQUIVALENT


def test_los_pesos_del_compuesto_si_estan_unificados():
    """Contraste deliberado: los pesos SÍ se comparten, porque los dos caminos
    llaman a `composite_score()` de `tte_formula` en vez de reimplementarla.
    Es el estado al que deberían llegar también las ocho constantes de arriba."""
    assert compartido.XWOBA_WEIGHT + compartido.BARREL_WEIGHT + compartido.PLATE_WEIGHT \
        == pytest.approx(1.0), "los pesos del compuesto deben sumar 1"
    assert vivo.composite_score is compartido.composite_score
    assert pit._shared_regress is compartido.regress
