# ============================================================================== 
# IMPORTACIONES
# ============================================================================== 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math, warnings

warnings.filterwarnings('ignore')

# ============================================================================== 
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================================== 
print("Cargando datos...")
X_data = pd.read_csv("flume_mvx_reflectance.csv")
Y_data_conventional = pd.read_csv("laboratory_measurements.csv", na_values="<LOQ")
Y_data_organic = pd.read_csv("laboratory_measurements_organic_chemicals.csv", na_values="<LOQ")

X_data['timestamp_iso'] = pd.to_datetime(X_data['timestamp_iso'])
Y_data_conventional['timestamp_iso'] = pd.to_datetime(Y_data_conventional['timestamp_iso'])
Y_data_organic['timestamp_iso'] = pd.to_datetime(Y_data_organic['timestamp_iso'])

# Merge
Y_data_completo = pd.merge(Y_data_conventional, Y_data_organic, on="timestamp_iso", how='outer')
datos_completos = pd.merge(X_data, Y_data_completo, on="timestamp_iso", how='inner')

# Variables espectrales
X = datos_completos.iloc[:, 2:202]  # columnas espectrales

# Contaminantes disponibles
contaminantes = [col for col in datos_completos.columns if col.startswith("lab_")]

Y_original = datos_completos[contaminantes].fillna(0)

# ============================================================================== 
# GENERAR CLASES BINARIAS (0=Bajo, 1=Alto) usando mediana
# ============================================================================== 
Y_binary = np.zeros_like(Y_original.values, dtype=int)
for i, col in enumerate(Y_original.columns):
    med = np.median(Y_original[col])
    Y_binary[:, i] = np.where(Y_original[col] <= med, 0, 1)

# ============================================================================== 
# TRAIN/TEST SPLIT
# ============================================================================== 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_binary, test_size=0.2, random_state=42)

# Escalar solo X
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

# Convertir a tensores
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)

# ============================================================================== 
# TRANSFORMER PARA CLASIFICACIÓN BINARIA MULTICOMPONENTE
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
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model*2, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.classifier = nn.Linear(d_model, num_outputs)  # salida binaria por contaminante
        self.num_outputs = num_outputs
    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = x.unsqueeze(0)  # (seq_len=1, batch, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # (batch, d_model)
        return torch.sigmoid(self.classifier(x))  # valores entre 0 y 1

# ============================================================================== 
# ENTRENAMIENTO
# ============================================================================== 
def train_model(model, X_tensor, Y_tensor, epochs=200, batch_size=32, lr=0.0005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    dataset = TensorDataset(X_tensor, Y_tensor.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_Y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_Y.float())
            loss.backward()
            optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")
    return model

# ============================================================================== 
# EVALUACIÓN
# ============================================================================== 
def evaluate_model(model, X_test, Y_test, contaminantes):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        preds = (logits.numpy() > 0.5).astype(int)
        results = {}
        for i, cont in enumerate(contaminantes):
            y_true = Y_test[:, i]
            y_pred = preds[:, i]
            try:
                auc = roc_auc_score(y_true, logits[:, i].numpy())
            except:
                auc = np.nan
            f1 = f1_score(y_true, y_pred)
            results[cont] = {"F1": f1, "AUC": auc, "ConfMatrix": confusion_matrix(y_true, y_pred)}
    return results

# ============================================================================== 
# EJECUCIÓN
# ============================================================================== 
model = SpectralTransformer(num_outputs=Y_train.shape[1])
model = train_model(model, X_train_tensor, Y_train_tensor, epochs=200)

metrics = evaluate_model(model, X_test_scaled, Y_test, contaminantes)

# Mostrar resultados
for cont, vals in metrics.items():
    print(f"\n--- {cont} ---")
    print(f"F1-score: {vals['F1']:.3f}, AUC: {vals['AUC']:.3f}")
    print("Matriz de confusión:")
    print(vals["ConfMatrix"])
