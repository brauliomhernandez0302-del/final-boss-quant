import pandas as pd

def procesar_datos_api(data):
    """
    Limpia y organiza los datos de cuotas recibidos desde la API.
    """
    if not data:
        return pd.DataFrame()

    # Ejemplo de transformación
    df = pd.DataFrame(data)
    df.columns = [col.strip().capitalize() for col in df.columns]
    df = df.drop_duplicates().dropna(how='all')

    return df
