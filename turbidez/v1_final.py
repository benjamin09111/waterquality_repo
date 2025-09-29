# ==============================================================================
# PASO 1: IMPORTACIONES Y CONFIGURACIÓN
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import warnings
import math

warnings.filterwarnings('ignore')

# ==============================================================================
# PASO 2: CARGAR Y PREPARAR DATOS
# ==============================================================================
print("Iniciando la carga y preparación de datos...")
try:
    X_data = pd.read_csv("flume_mvx_reflectance.csv")
    Y_data = pd.read_csv("flume_turbimax_turbidity.csv")
except FileNotFoundError:
    print("Error: Asegúrate de que los archivos 'flume_mvx_reflectance.csv' y 'flume_turbimax_turbidity.csv' estén en la misma carpeta.")
    exit()

X_data['timestamp_iso'] = pd.to_datetime(X_data['timestamp_iso'])
Y_data['timestamp_iso'] = pd.to_datetime(Y_data['timestamp_iso'])
Y_data = Y_data[Y_data['valid_data'] == 1]
datos_completos = pd.merge(X_data, Y_data, on="timestamp_iso", how='inner')

X = datos_completos.iloc[:, 2:202].values
Y = datos_completos['turbimax_turbidity_fnu'].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

print(f"Datos listos. Tamaño del set de entrenamiento: {len(X_train_tensor)} muestras.")

# ==============================================================================
# PASO 3: DEFINIR LA ARQUITECTURA DEL TRANSFORMER (Versión Simple)
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SpectralTransformer(nn.Module):
    # Arquitectura simple
    def __init__(self, input_dim=200, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super(SpectralTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = x.unsqueeze(0)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        output = self.regressor(x)
        return output

model = SpectralTransformer(input_dim=200, d_model=32, nhead=4, num_layers=2, dropout=0.1)
print("\nModelo Transformer creado (versión simple):")
print(model)

# ==============================================================================
# PASO 4: ENTRENAR EL MODELO
# ==============================================================================
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
epochs = 200

print("\nIniciando entrenamiento...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        prediction = model(batch_X)
        loss = loss_function(prediction, batch_Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Pérdida Promedio: {avg_loss:.4f}")

print("Entrenamiento finalizado.")

# ==============================================================================
# PASO 5: EVALUAR Y VER LOS RESULTADOS
# ==============================================================================
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).squeeze()
    Y_test_squeezed = Y_test_tensor.squeeze()
    
    test_loss = loss_function(test_predictions, Y_test_squeezed)
    print(f"\nPérdida (MSE) en el conjunto de prueba: {test_loss.item():.4f}")

    print("\n--- Muestra de Predicciones Individuales ---")
    num_muestras = 10
    for i in range(num_muestras):
        prediccion = test_predictions[i].item()
        real = Y_test_squeezed[i].item()
        diferencia = abs(prediccion - real)
        print(f"\n[Muestra de prueba #{i+1}]\n  > Predicción: {prediccion:.2f}\n  > Valor Real: {real:.2f}\n  > Diferencia: {diferencia:.2f}")
    print("\n-------------------------------------------")

    plt.figure(figsize=(8, 8))
    plt.scatter(Y_test_tensor.numpy(), test_predictions.numpy(), alpha=0.5)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], '--r', linewidth=2)
    plt.xlabel("Valores Reales de Turbidez")
    plt.ylabel("Predicciones del Modelo")
    plt.title("Comparación de Valores Reales vs. Predicciones")
    plt.grid(True)
    plt.show()

# ==============================================================================
# MÉTRICAS DE REGRESIÓN ADICIONALES
# ==============================================================================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = Y_test_tensor.numpy().squeeze()
y_pred = test_predictions.numpy().squeeze()

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n--- Métricas de regresión ---")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")

# ==============================================================================
# PASO 6: MÉTRICAS DE CLASIFICACIÓN (COMPARACIÓN CON PAPER)
# ==============================================================================
from sklearn.metrics import f1_score, roc_auc_score

umbral_bajo = 50.0
umbral_alto = 150.0
print(f"\nUsando los umbrales del paper para clasificación: Bajo (<{umbral_bajo}) y Alto (>{umbral_alto}).")

indices_validos = np.where((y_true < umbral_bajo) | (y_true > umbral_alto))
Y_test_filtrado = y_true[indices_validos]
predicciones_filtradas = y_pred[indices_validos]

Y_test_clases = (Y_test_filtrado > umbral_alto).astype(int)
predicciones_clases = (predicciones_filtradas > umbral_alto).astype(int)

if len(np.unique(Y_test_clases)) > 1:
    f1 = f1_score(Y_test_clases, predicciones_clases)
    auc = roc_auc_score(Y_test_clases, predicciones_filtradas)
    print(f"\nF1-Score (método paper): {f1:.4f}")
    print(f"AUC (método paper): {auc:.4f}")

    print("\n--- Comparación Final con el Paper (para Turbidez) ---")
    print(f"Métrica      | Tu Transformer | Paper (LSTM)")
    print(f"-------------|----------------|---------------")
    print(f"F1-Score     | {f1:.4f}         | 0.5490")
    print(f"AUC          | {auc:.4f}         | 0.6880")
    print("-------------------------------------------------")
else:
    print("El conjunto de prueba filtrado no contiene muestras de ambas clases (Bajo y Alto), por lo que no se pueden calcular F1/AUC.")
