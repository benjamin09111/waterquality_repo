# ==============================================================================  
# IMPORTS  
# ==============================================================================  
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score, roc_auc_score
import math
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================  
# CARGA DE DATOS  
# ==============================================================================  
print("Cargando datos...")
X_data = pd.read_csv("flume_mvx_reflectance.csv")
Y_data_conventional = pd.read_csv("laboratory_measurements.csv", na_values=["<LOQ", "<LOQ>"])
Y_data_organic = pd.read_csv("laboratory_measurements_organic_chemicals.csv", na_values=["<LOQ", "<LOQ>"])

X_data['timestamp_iso'] = pd.to_datetime(X_data['timestamp_iso'])
Y_data_conventional['timestamp_iso'] = pd.to_datetime(Y_data_conventional['timestamp_iso'])
Y_data_organic['timestamp_iso'] = pd.to_datetime(Y_data_organic['timestamp_iso'])

Y_data_completo = pd.merge(Y_data_conventional, Y_data_organic, on="timestamp_iso", how='outer')
datos_completos = pd.merge(X_data, Y_data_completo, on="timestamp_iso", how='inner')

X = datos_completos.iloc[:, 2:202]

# Contaminantes
contaminantes_mg_l = [
    'lab_turbidity_ntu', 'lab_tss_mg_l', 'lab_doc_mg_l', 'lab_toc_mg_l', 'lab_nsol_mg_l',
    'lab_ntot_mg_l', 'lab_nh4_mg_l', 'lab_po4_mg_l', 'lab_so4_mg_l'
]
contaminantes_ng_l = [
    'lab_acesulfame_ng_l', 'lab_caffeine_ng_l', 'lab_cyclamate_ng_l', 'lab_candesartan_ng_l',
    'lab_citalopram_ng_l', 'lab_diclofenac_ng_l', 'lab_hydrochlorthiazide_ng_l',
    'lab_triclosan_ng_l', 'lab_13-diphenylguanidine_ng_l', 'lab_6ppd-quinone_ng_l',
    'lab_hmmm_ng_l', 'lab_24-d_ng_l', 'lab_carbendazim_ng_l', 'lab_diuron_ng_l',
    'lab_mcpa_ng_l', 'lab_mecoprop_ng_l', 'lab_oit_ng_l', 'lab_4-&5-methylbenzotriazole_ng_l',
    'lab_benzotriazole_ng_l', 'lab_deet_ng_l'
]

contaminantes_mg_l = [col for col in contaminantes_mg_l if col in datos_completos.columns]
contaminantes_ng_l = [col for col in contaminantes_ng_l if col in datos_completos.columns]

# Valores
Y_mg_l_original = datos_completos[contaminantes_mg_l].fillna(0)
Y_ng_l_original = datos_completos[contaminantes_ng_l].fillna(0).astype(float)

# Escalado log para ng/L
Y_ng_l_log = np.log1p(Y_ng_l_original.values)

# Split
X_train, X_test, \
Y_mg_l_train, Y_mg_l_test, \
Y_ng_l_train, Y_ng_l_test, \
Y_ng_l_original_train, Y_ng_l_original_test = train_test_split(
    X, Y_mg_l_original.values, Y_ng_l_log, Y_ng_l_original.values, test_size=0.2, random_state=42
)

# Escalado
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

scaler_y_mg = StandardScaler()
Y_mg_l_train_scaled = scaler_y_mg.fit_transform(Y_mg_l_train)

scaler_y_ng = StandardScaler()
Y_ng_l_train_scaled = scaler_y_ng.fit_transform(Y_ng_l_train)

# Tensores
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
Y_mg_l_train_tensor = torch.tensor(Y_mg_l_train_scaled, dtype=torch.float32)
Y_ng_l_train_tensor = torch.tensor(Y_ng_l_train_scaled, dtype=torch.float32)

# ==============================================================================  
# DEFINICIÓN DEL MODELO  
# ==============================================================================  
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0)]

class SpectralTransformer(nn.Module):
    def __init__(self, num_outputs, input_dim=200, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.regressor = nn.Linear(d_model, num_outputs)
    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = x.unsqueeze(0)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        return self.regressor(x)

def train_model(model, train_loader, epochs=100, lr=0.0005, model_name=""):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"\n--- Entrenando Modelo para {model_name} ---")
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            prediction = model(batch_X)
            loss = loss_function(prediction, batch_Y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} completada.")
    print(f"Entrenamiento de modelo {model_name} finalizado.")
    return model

# ==============================================================================  
# ENTRENAMIENTO  
# ==============================================================================  
model_mg_l = SpectralTransformer(num_outputs=Y_mg_l_train.shape[1])
train_loader_mg_l = DataLoader(TensorDataset(X_train_tensor, Y_mg_l_train_tensor), batch_size=32, shuffle=True)
model_mg_l = train_model(model_mg_l, train_loader_mg_l, epochs=200, model_name="mg/L")

model_ng_l = SpectralTransformer(num_outputs=Y_ng_l_train.shape[1])
train_loader_ng_l = DataLoader(TensorDataset(X_train_tensor, Y_ng_l_train_tensor), batch_size=32, shuffle=True)
model_ng_l = train_model(model_ng_l, train_loader_ng_l, epochs=500, model_name="ng/L")

# ==============================================================================  
# PREDICCIÓN  
# ==============================================================================  
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# mg/L
model_mg_l.eval()
with torch.no_grad():
    pred_mg_l_scaled = model_mg_l(X_test_tensor).numpy()
    pred_mg_l = scaler_y_mg.inverse_transform(pred_mg_l_scaled)

# ng/L
model_ng_l.eval()
with torch.no_grad():
    pred_ng_l_scaled = model_ng_l(X_test_tensor).numpy()
    pred_ng_l_log = scaler_y_ng.inverse_transform(pred_ng_l_scaled)
    pred_ng_l = np.expm1(pred_ng_l_log)

# ==============================================================================  
# MÉTRICAS DE REGRESIÓN  
# ==============================================================================  
print("\n--- Métricas de Regresión ---")
def metricas_regresion(y_true, y_pred, nombres):
    for i, col in enumerate(nombres):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"{col}: R²={r2:.3f}, MSE={mse:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

metricas_regresion(Y_mg_l_test, pred_mg_l, contaminantes_mg_l)
metricas_regresion(Y_ng_l_original_test, pred_ng_l, contaminantes_ng_l)

# ==============================================================================  
# CLASIFICACIÓN POR PERCENTILES  
# ==============================================================================  
def clasificar_por_percentil(valores, datos_originales):
    clasificaciones = np.zeros_like(valores, dtype=int)
    for i in range(valores.shape[1]):
        serie = datos_originales[:, i]
        perc33 = np.percentile(serie, 33)
        perc66 = np.percentile(serie, 66)
        clasificaciones[:, i] = np.where(valores[:, i] < perc33, 0,
                                 np.where(valores[:, i] < perc66, 1, 2))
    return clasificaciones

# Clasificaciones
y_true_mg_class = clasificar_por_percentil(Y_mg_l_test, Y_mg_l_test)
y_pred_mg_class = clasificar_por_percentil(pred_mg_l, Y_mg_l_test)

y_true_ng_class = clasificar_por_percentil(Y_ng_l_original_test, Y_ng_l_original_test)
y_pred_ng_class = clasificar_por_percentil(pred_ng_l, Y_ng_l_original_test)

# ==============================================================================  
# MÉTRICAS DE CLASIFICACIÓN  
# ==============================================================================  
print("\n--- Métricas de Clasificación ---")
def metricas_clasificacion(y_true, y_pred, nombres):
    for i, col in enumerate(nombres):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        f1 = f1_score(y_true[:, i], y_pred[:, i], average="weighted")
        y_true_bin = label_binarize(y_true[:, i], classes=[0,1,2])
        y_pred_bin = label_binarize(y_pred[:, i], classes=[0,1,2])
        try:
            auc = roc_auc_score(y_true_bin, y_pred_bin, average="weighted", multi_class="ovr")
        except:
            auc = np.nan
        print(f"{col}: Accuracy={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")

metricas_clasificacion(y_true_mg_class, y_pred_mg_class, contaminantes_mg_l)
metricas_clasificacion(y_true_ng_class, y_pred_ng_class, contaminantes_ng_l)
