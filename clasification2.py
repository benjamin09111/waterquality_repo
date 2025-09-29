# mejor_transformer_pipeline.py
# Versión mejorada: Transformer secuencial + pipeline jerárquico (detección -> baja/alta)
# Ejecutar en Python 3.8+ con torch, sklearn, pandas, numpy

import os
import math
import random
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# -------------------------
# CARGA Y PREPARACIÓN (igual a tu flujo)
# -------------------------
X_data = pd.read_csv("flume_mvx_reflectance.csv")
Y_data_conventional = pd.read_csv("laboratory_measurements.csv", na_values="<LOQ")
Y_data_organic = pd.read_csv("laboratory_measurements_organic_chemicals.csv", na_values="<LOQ")

# parse timestamps (ya lo tenías)
X_data['timestamp_iso'] = pd.to_datetime(X_data['timestamp_iso'])
Y_data_conventional['timestamp_iso'] = pd.to_datetime(Y_data_conventional['timestamp_iso'])
Y_data_organic['timestamp_iso'] = pd.to_datetime(Y_data_organic['timestamp_iso'])

# merge igual que antes (inner join entre X y Y por timestamp)
Y_data_completo = pd.merge(Y_data_conventional, Y_data_organic, on="timestamp_iso", how='outer')
datos_completos = pd.merge(X_data, Y_data_completo, on="timestamp_iso", how='inner')

# detect spectral columns (ajusta si cambian posiciones)
# asumiendo que en tu csv X espectral ocupaba columnas 2:202 (exactamente 200 bandas)
spectral_cols = datos_completos.columns[2:202]
X = datos_completos[spectral_cols].astype(float).reset_index(drop=True)

# contaminantes (tus columnas lab_)
contaminantes = [c for c in datos_completos.columns if c.startswith("lab_")]
Y_original = datos_completos[contaminantes].fillna(0).reset_index(drop=True)

print("Samples:", X.shape[0], "Bands:", X.shape[1], "Contaminantes:", len(contaminantes))

# -------------------------
# DEFINIR LABELS: etapa 1 (presencia detectable) y etapa 2 (baja/alta)
# - presencia: valor > 0 (sigues usando los datos tal cual)
# - level (baja/alta): usar mediana entre los detectables (por contaminante)
# -------------------------
Y_values = Y_original.values  # shape (N, M)
presence = (Y_values > 0).astype(int)  # 1 si detectado (>0), 0 si no detectable

# Para level: solo tiene sentido donde presence==1. Defino umbral = mediana de los detectables por contaminante.
num_samples, num_conts = Y_values.shape
level_labels = np.zeros_like(Y_values, dtype=int)  # 0 = baja, 1 = alta (por contaminante)
for j in range(num_conts):
    vals_pos = Y_values[presence[:, j] == 1, j]
    if vals_pos.size == 0:
        # no detectables -> deja todo como 0 (baja) por defecto
        thresh = 0.0
    else:
        thresh = np.median(vals_pos)
    # para los detectables, asigna segun mediana
    idx_pos = np.where(presence[:, j] == 1)[0]
    if idx_pos.size > 0:
        level_labels[idx_pos, j] = (Y_values[idx_pos, j] > thresh).astype(int)

# Ahora tenemos:
# - presence: (N, M) binario
# - level_labels: (N, M) binario (solo confiable donde presence==1)

# -------------------------
# TRAIN/TEST SPLIT (misma lógica tuya)
# -------------------------
X_train, X_test, pres_train, pres_test, lvl_train, lvl_test = train_test_split(
    X.values, presence, level_labels, test_size=0.2, random_state=SEED
)

# Escalado: escala por banda (StandardScaler) - fit solo en train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir a tensores
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
P_train_t = torch.tensor(pres_train, dtype=torch.float32)  # presence
P_test_t = torch.tensor(pres_test, dtype=torch.float32)
L_train_t = torch.tensor(lvl_train, dtype=torch.float32)  # level (baja/alta)
L_test_t = torch.tensor(lvl_test, dtype=torch.float32)

# -------------------------
# DATALOADERS
# -------------------------
BATCH_SIZE = 32
train_ds = TensorDataset(X_train_t, P_train_t, L_train_t)
test_ds = TensorDataset(X_test_t, P_test_t, L_test_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# TRANSFORMER MODEL (por token: cada banda = token)
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class SpectralTransformerHierarchical(nn.Module):
    def __init__(self, seq_len, num_outputs, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.num_outputs = num_outputs
        self.d_model = d_model

        # embed each scalar reflectance (per band) to d_model
        self.token_embed = nn.Linear(1, d_model)  # input per token is scalar
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Heads (logits). We'll produce per-contaminant logits.
        self.presence_head = nn.Linear(d_model, num_outputs)   # logits for presence (sigmoid)
        self.level_head = nn.Linear(d_model, num_outputs)      # logits for low/high (sigmoid)

        # init
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.presence_head.weight)
        nn.init.xavier_uniform_(self.level_head.weight)

    def forward(self, x):
        # x: (batch, seq_len)
        b = x.size(0)
        seq_len = x.size(1)
        # shape to (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        x = self.token_embed(x) * math.sqrt(self.d_model)  # (batch, seq_len, d_model)

        # prepend CLS
        cls = self.cls_token.expand(b, -1, -1)  # (batch,1,d_model)
        x = torch.cat([cls, x], dim=1)  # (batch, seq_len+1, d_model)
        x = self.pos_encoder(x)

        # transformer: returns (batch, seq_len+1, d_model)
        x_enc = self.transformer(x)

        # take CLS embedding for global representation
        cls_emb = x_enc[:, 0, :]  # (batch, d_model)

        presence_logits = self.presence_head(cls_emb)  # (batch, M)
        level_logits = self.level_head(cls_emb)        # (batch, M)

        return presence_logits, level_logits

# -------------------------
# Entrenamiento
# -------------------------
model = SpectralTransformerHierarchical(seq_len=X_train_t.shape[1], num_outputs=num_conts,
                                       d_model=128, nhead=4, num_layers=3, dropout=0.1)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
pres_loss_fn = nn.BCEWithLogitsLoss(reduction="none")  # we'll handle masking ourselves
level_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

EPOCHS = 120

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for batch_X, batch_P, batch_L in train_loader:
        batch_X = batch_X.to(DEVICE)
        batch_P = batch_P.to(DEVICE)
        batch_L = batch_L.to(DEVICE)

        optimizer.zero_grad()
        pres_logits, lvl_logits = model(batch_X)

        # presence loss (per element)
        pres_loss_all = pres_loss_fn(pres_logits, batch_P)  # (batch, M)
        pres_loss = pres_loss_all.mean()  # average over all elements

        # level loss must be computed only where presence==1
        lvl_loss_all = level_loss_fn(lvl_logits, batch_L)  # (batch, M)
        mask = batch_P  # mask = 1 where presence
        # sum masked loss and divide by number of positives to avoid scale issues
        denom = mask.sum()
        if denom.item() > 0:
            lvl_loss = (lvl_loss_all * mask).sum() / denom
        else:
            lvl_loss = torch.tensor(0.0, device=DEVICE)

        loss = pres_loss + lvl_loss  # equal weighting
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / len(train_ds)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{EPOCHS} - loss: {epoch_loss:.5f}")

# -------------------------
# Evaluación: calcular métricas por contaminante
# -------------------------
model.eval()
with torch.no_grad():
    # get logits on test
    X_test_dev = X_test_t.to(DEVICE)
    pres_logits_all, lvl_logits_all = model(X_test_dev)
    pres_probs = torch.sigmoid(pres_logits_all).cpu().numpy()  # (n_test, M)
    lvl_probs = torch.sigmoid(lvl_logits_all).cpu().numpy()

    pres_preds = (pres_probs >= 0.5).astype(int)
    lvl_preds = (lvl_probs >= 0.5).astype(int)

    # métricas por contaminante
    results_detect: Dict[str, Dict] = {}
    results_level: Dict[str, Dict] = {}

    for j, cont in enumerate(contaminantes):
        y_true_pres = pres_test[:, j]  # numpy
        y_pred_pres = pres_preds[:, j]
        y_score_pres = pres_probs[:, j]

        # detection metrics
        try:
            auc_pres = roc_auc_score(y_true_pres, y_score_pres)
        except ValueError:
            auc_pres = np.nan
        f1_pres = f1_score(y_true_pres, y_pred_pres, zero_division=0)
        cm_pres = confusion_matrix(y_true_pres, y_pred_pres)

        results_detect[cont] = {"F1": float(f1_pres), "AUC": float(auc_pres), "ConfMatrix": cm_pres.tolist()}

        # level metrics: evaluate ONLY where true presence == 1
        idx_pos = np.where(pres_test[:, j] == 1)[0]
        if idx_pos.size == 0:
            # no positive samples in test for this contaminante
            results_level[cont] = {"F1": None, "AUC": None, "N_pos_test": 0}
        else:
            y_true_lvl = lvl_test[idx_pos, j]
            y_pred_lvl = lvl_preds[idx_pos, j]
            y_score_lvl = lvl_probs[idx_pos, j]
            try:
                auc_lvl = roc_auc_score(y_true_lvl, y_score_lvl)
            except ValueError:
                auc_lvl = np.nan
            f1_lvl = f1_score(y_true_lvl, y_pred_lvl, zero_division=0)
            cm_lvl = confusion_matrix(y_true_lvl, y_pred_lvl)
            results_level[cont] = {"F1": float(f1_lvl), "AUC": float(auc_lvl), "ConfMatrix": cm_lvl.tolist(), "N_pos_test": int(idx_pos.size)}

# -------------------------
# Mostrar resultados
# -------------------------
print("\n--- DETECCIÓN (presencia) por contaminante ---")
for cont, vals in results_detect.items():
    print(f"{cont}: F1={vals['F1']:.3f}, AUC={np.nan if np.isnan(vals['AUC']) else vals['AUC']:.3f}, CM={vals['ConfMatrix']}")

print("\n--- CLASIFICACIÓN (baja/alta) por contaminante (solo sobre muestras detectadas en test) ---")
for cont, vals in results_level.items():
    if vals["N_pos_test"] == 0:
        print(f"{cont}: SIN POSITIVOS EN TEST (N_pos_test=0) — no es posible evaluar level.")
    else:
        print(f"{cont}: F1={vals['F1']:.3f}, AUC={np.nan if np.isnan(vals['AUC']) else vals['AUC']:.3f}, N_pos_test={vals['N_pos_test']}, CM={vals['ConfMatrix']}")

# Opcional: guardar métricas a CSV para inclusion en el paper
df_detect = pd.DataFrame.from_dict(results_detect, orient="index")
df_level = pd.DataFrame.from_dict(results_level, orient="index")
df_detect.to_csv("metrics_presence_by_contaminant.csv")
df_level.to_csv("metrics_level_by_contaminant.csv")

print("\nMétricas guardadas: metrics_presence_by_contaminant.csv, metrics_level_by_contaminant.csv")
