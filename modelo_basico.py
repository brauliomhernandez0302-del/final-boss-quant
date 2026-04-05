








import pandas as pd
from sklearn.linear_model import LogisticRegression

# ===== Datos de ejemplo (partidos pasados) =====
# En un futuro esto lo jalaremos de internet o archivos reales
# Por ahora creamos un dataset pequeño de ejemplo

data = {
"puntos_totales": [180, 195, 200, 175, 210, 185, 220, 178, 199, 205],
"over_180": [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
# 1 = se dio el over 180, 0 = no se dio
}

df = pd.DataFrame(data)

# ===== Modelo simple =====
X = df[["puntos_totales"]] # Variable de entrada
y = df["over_180"] # Resultado (Over/Under)

modelo = LogisticRegression()
modelo.fit(X, y)

# ===== Predicción =====
nuevo_partido = [[190]] # Supón que el total proyectado es 190
prediccion = modelo.predict(nuevo_partido)

print("Predicción para Over 180 con 190 puntos proyectados:", "Over ✅" if prediccion[0]==1 else "Under ❌")
