import pandas as pd
import numpy as np

# Archivos
file1 = "laboratory_measurements.csv"
file2 = "laboratory_measurements_organic_chemicals.csv"

# Cargar datasets, interpretando <LOQ como NaN
df1 = pd.read_csv(file1, na_values="<LOQ>")
df2 = pd.read_csv(file2, na_values="<LOQ>")

def class_balance_analysis(df):
    results = []
    for col in df.filter(like="lab_").columns:
        # Convertir a numérico (forzando no numéricos a NaN)
        vals = pd.to_numeric(df[col], errors="coerce")
        
        # Solo detectables (>0)
        detectables = vals[vals > 0].dropna()
        
        if len(detectables) == 0:
            results.append({
                "contaminante": col,
                "total_detectables": 0,
                "baja_count": 0,
                "alta_count": 0,
                "baja_pct": np.nan,
                "alta_pct": np.nan
            })
            continue
        
        # Mediana entre detectables
        mediana = detectables.median()
        
        # Clasificación baja/alta
        baja = (detectables <= mediana).sum()
        alta = (detectables > mediana).sum()
        
        total = baja + alta
        results.append({
            "contaminante": col,
            "total_detectables": total,
            "baja_count": baja,
            "alta_count": alta,
            "baja_pct": round(baja / total * 100, 1),
            "alta_pct": round(alta / total * 100, 1)
        })
    
    return pd.DataFrame(results)

# Ejecutar análisis
analysis1 = class_balance_analysis(df1)
analysis2 = class_balance_analysis(df2)

print("\n--- Balance de clases (conventional) ---\n")
print(analysis1)

print("\n--- Balance de clases (organic chemicals) ---\n")
print(analysis2)

# Opcional: guardar a CSV
analysis1.to_csv("class_balance_conventional.csv", index=False)
analysis2.to_csv("class_balance_organic.csv", index=False)
