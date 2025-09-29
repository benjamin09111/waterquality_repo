# este codigo tiene como objetivo simular el C2, de clasificación, pero solo enfocado en turbidez. Esto es para el caso que tuvieramos muchos mas datos.

# ============================================================================== 
# C2 PARA TURBIDEZ (CON MÁS DATOS)
# ==============================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
import math
import warnings

warnings.filterwarnings("ignore")

# ============================================================================== 
# PASO 1: CARGAR DATOS
# ==============================================================================
X_data = pd.read_csv("flume_mvx_reflectance.csv")  # reflectancia MVX
Y_data = pd.read_csv("flume_turbimax_turbidity.csv")  # turbidez
Y_data = Y_data[Y_data['valid_data'] == 1]  # solo datos válidos

# Convertir timestamps
X_data['timestamp_iso'] = pd.to_datetime(X_data['timestamp_iso'])
Y_data['timestamp_iso'] = pd.to_datetime(Y_data['timestamp_iso'])

# Merge
datos_completos = pd.merge(X_data, Y_data, on="timestamp_iso", how='inner')

# Features y target
X = datos_completos.iloc[:, 2:202].values  # reflectancia
Y = datos_completos['turbimax_turbidity_fnu'].values.reshape(-1, 1)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tensores
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# ============================================================================== 
# PASO 2: DEFINIR TRANSFORMER AL ESTILO C2
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:,0,0::2] = torch.sin(position * div_term)
        pe[:,0,1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SpectralTransformer(nn.Module):
    def __init__(self, input_dim=200, d_model=64, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.regressor = nn.Linear(d_model, 1)
        self.d_model = d_model
        
    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = x.unsqueeze(0)  # shape: [seq_len, batch, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        output = self.regressor(x)
        return output

# Crear modelo
model = SpectralTransformer(input_dim=200, d_model=64, nhead=8, num_layers=3, dropout=0.1)
print(model)

# ============================================================================== 
# PASO 3: ENTRENAMIENTO
# ==============================================================================
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
epochs = 200

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = loss_function(pred, batch_Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss Promedio: {total_loss/len(train_loader):.4f}")

# ============================================================================== 
# PASO 4: EVALUACIÓN
# ==============================================================================
model.eval()
with torch.no_grad():
    test_pred = model(X_test_tensor).squeeze()
    Y_test_squeezed = Y_test_tensor.squeeze()
    mse_test = loss_function(test_pred, Y_test_squeezed)
    print(f"\nMSE en conjunto de prueba: {mse_test.item():.4f}")
    
    # Scatter plot
    plt.figure(figsize=(8,6))
    plt.scatter(Y_test_tensor.numpy(), test_pred.numpy(), alpha=0.5)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], '--r', linewidth=2)
    plt.xlabel("Valores reales de Turbidez")
    plt.ylabel("Predicciones del Transformer")
    plt.title("Valores reales vs predicciones")
    plt.grid(True)
    plt.show()

# ============================================================================== 
# PASO 5: CLASIFICACIÓN (BAJA/ALTA) PARA F1 Y AUC
# ==============================================================================
umbral_bajo = 50.0
umbral_alto = 150.0

Y_test_np = Y_test_tensor.numpy().squeeze()
pred_np = test_pred.numpy().squeeze()
indices_validos = np.where((Y_test_np < umbral_bajo) | (Y_test_np > umbral_alto))

Y_test_filtro = Y_test_np[indices_validos]
pred_filtro = pred_np[indices_validos]

Y_test_clases = (Y_test_filtro > umbral_alto).astype(int)
pred_clases = (pred_filtro > umbral_alto).astype(int)

if len(np.unique(Y_test_clases)) > 1:
    f1 = f1_score(Y_test_clases, pred_clases)
    auc = roc_auc_score(Y_test_clases, pred_filtro)
    print(f"\nF1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
else:
    print("No hay muestras de ambas clases para calcular F1/AUC.")
