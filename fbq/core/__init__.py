"""fbq.core — paso 0: el contrato de tiempo e identidad.

No depende de nada. Todo lo demás depende de esto.

Su razón de ser: en el proyecto anterior las reglas de tiempo vivían repartidas
en cada llamador. `PITCache.get_latest()` comparaba con `<=` inclusivo y era
seguro sólo porque cuatro funciones distintas le restaban un segundo al día a
mano — hasta que apareció una quinta que no lo hacía. El día del juego se
derivaba a veces del `officialDate` del schedule y a veces del timestamp UTC
truncado, que para cualquier nocturno de la costa oeste cae un día adelante.

Acá las reglas son funciones, no disciplina.
"""

from fbq.core.clock import (AsOf, cutoff_para_dia, dia_oficial,
                            instantanea_vigente, es_anterior,
                            normalizar_utc, tiene_hora)
from fbq.core.identity import (ESTADOS_FINALES, MARGEN_MINIMO,
                               VENTANA_EMPAREJAMIENTO, elegir_unico,
                               mismo_equipo)

__all__ = [
    "AsOf", "cutoff_para_dia", "dia_oficial", "instantanea_vigente",
    "es_anterior", "normalizar_utc", "tiene_hora", "ESTADOS_FINALES", "MARGEN_MINIMO", "VENTANA_EMPAREJAMIENTO",
    "elegir_unico", "mismo_equipo",
]
