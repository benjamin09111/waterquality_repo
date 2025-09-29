import pandas as pd

# Rutas de los archivos (ajusta si están en otra carpeta)
file1 = "laboratory_measurements.csv"
file2 = "laboratory_measurements_organic_chemicals.csv"

# Cargar los datasets
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Función para analizar un dataset (versión corregida)
def analyze_dataset(df):
    results = {}
    # Itera solo sobre las columnas que nos interesan (las de laboratorio)
    for col in df.filter(like='lab_').columns:
        
        # --- LÍNEA CORREGIDA ---
        # Primero, convertimos la columna a texto para poder usar .contains() sin errores.
        # El .str.strip() elimina espacios en blanco ocultos (ej: ' <LOQ> ')
        # na=False asegura que los valores vacíos (NaN) no den problemas.
        es_loq = df[col].astype(str).str.strip().str.contains("LOQ", na=False)
        
        # Para los ceros, convertimos a número (lo que no es número se vuelve NaN)
        # y comparamos con 0.
        es_cero = (pd.to_numeric(df[col], errors='coerce') == 0)
        
        # Sumamos las dos condiciones con el operador | (OR)
        zeros_o_loq = (es_cero | es_loq).sum()
        # ------------------------

        results[col] = {
            "total_no_vacios": int(df[col].notna().sum()),
            "ceros_o_LOQ": int(zeros_o_loq),
            "faltantes": int(df[col].isna().sum()),
            "total_registros": int(len(df))
        }
    return pd.DataFrame(results).T

# Analizar ambos datasets
analysis1 = analyze_dataset(df1)
analysis2 = analyze_dataset(df2)

# Mostrar resultados
print("\n--- Análisis laboratory_measurements.csv ---\n")
print(analysis1)

print("\n--- Análisis laboratory_measurements_organic_chemicals.csv ---\n")
print(analysis2)