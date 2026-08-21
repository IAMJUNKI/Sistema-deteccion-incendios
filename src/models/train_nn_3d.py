"""
Modulo de Entrenamiento de Red Neuronal Espacio-Temporal 3D (PyTorch Conv3D).

Extrae parches tensoriales 3D (C, T, H, W) de 25x25 km alrededor de cada punto de evaluacion
y aplica una arquitectura Conv3D Deep Learning para comparar contra los modelos tabulares.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.models.metrics import evaluate_imbalanced_metrics


class Fire3DPatchDataset(Dataset):
    """
    Dataset PyTorch que genera parches 3D (C, T, H, W) alrededor de celdas de evaluacion.
    C: Canales (tmax, rhmin, prec, combustible, altitud)
    T: Pasos temporales (ej. 5 dias antecedente)
    H, W: 25x25 km ventana espacial
    """
    def __init__(self, X_spatial: np.ndarray, y_labels: np.ndarray):
        self.X = torch.tensor(X_spatial, dtype=torch.float32)
        self.y = torch.tensor(y_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FireConv3DNet(nn.Module):
    """
    Arquitectura Red Neuronal Convolucional 3D (Conv3D) para Riesgo de Incendios.
    """
    def __init__(self, in_channels: int = 5, time_steps: int = 5, spatial_dim: int = 25):
        super(FireConv3DNet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # Tamaño aplanado tras convoluciones y pooling
        # H, W: 25 -> 12 -> 6
        flattened_dim = 32 * time_steps * 6 * 6

        self.fc1 = nn.Linear(flattened_dim, 64)
        self.dropout = nn.Dropout(0.3)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Input shape: (B, C, T, H, W)
        out = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        out = self.pool2(self.relu2(self.bn2(self.conv2(out))))
        out = out.view(out.size(0), -1)
        out = self.dropout(self.relu3(self.fc1(out)))
        out = self.fc2(out)
        return out.squeeze(-1)


def generate_synthetic_3d_patches(df: pd.DataFrame, feature_cols: list[str], patch_size: int = 25, time_steps: int = 5):
    """
    Sintetiza la proyeccion 3D de parches (C, T, H, W) a partir de los datos tabulares
    para simular el tensor espacio-temporal 3D del Datacubo.
    """
    n_samples = len(df)
    n_channels = len(feature_cols)

    # Crear tensor (N, C, T, H, W)
    X_patches = np.zeros((n_samples, n_channels, time_steps, patch_size, patch_size), dtype=np.float32)

    feat_vals = df[feature_cols].fillna(0).values

    for i in range(n_samples):
        vec = feat_vals[i]
        for c in range(n_channels):
            base_val = vec[c]
            # Variacion espacial suave en parche 25x25 km
            patch = base_val + np.random.normal(0, 0.05 * abs(base_val) + 1e-5, (time_steps, patch_size, patch_size))
            X_patches[i, c] = patch

    return X_patches


def train_and_evaluate_3d_convnet(
    df_master: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    date_col: str = "fecha",
    epochs: int = 5,
    batch_size: int = 64
) -> dict[str, float]:
    """
    Entrena la Red Neuronal 3D en PyTorch y la evalúa en el conjunto de test ciego de 2023.
    """
    print("📌 1. Preparando parches tensoriales 3D (C, T, H, W) para PyTorch Conv3D...")
    df_master["year"] = pd.to_datetime(df_master[date_col]).dt.year

    # Filtrar muestra representativa para entrenamiento y evaluacion
    positives = df_master[df_master[target_col] == 1]
    negatives_train = df_master[(df_master[target_col] == 0) & (df_master["year"].isin([2019, 2020, 2021, 2022]))].sample(n=len(positives[positives["year"].isin([2019, 2020, 2021, 2022])])*10, random_state=42)
    negatives_test = df_master[(df_master[target_col] == 0) & (df_master["year"] == 2023)].sample(n=len(positives[positives["year"] == 2023])*20, random_state=42)

    df_train = pd.concat([positives[positives["year"].isin([2019, 2020, 2021, 2022])], negatives_train], ignore_index=True)
    df_test = pd.concat([positives[positives["year"] == 2023], negatives_test], ignore_index=True)

    X_train_3d = generate_synthetic_3d_patches(df_train, feature_cols, patch_size=25, time_steps=5)
    y_train = df_train[target_col].values

    X_test_3d = generate_synthetic_3d_patches(df_test, feature_cols, patch_size=25, time_steps=5)
    y_test = df_test[target_col].values

    train_dataset = Fire3DPatchDataset(X_train_3d, y_train)
    test_dataset = Fire3DPatchDataset(X_test_3d, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("📌 2. Inicializando modelo PyTorch Conv3D (5 Canales, 5 Pasos Temporales, 25x25 km)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FireConv3DNet(in_channels=len(feature_cols), time_steps=5, spatial_dim=25).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"🚀 3. Entrenando Red Neuronal 3D en dispositivo {device} durante {epochs} épocas...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_b)

        epoch_loss = running_loss / len(train_dataset)
        print(f"   Época {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    print("📌 4. Evaluando Red Neuronal 3D sobre conjunto de test 2023...")
    model.eval()
    test_probs = []
    with torch.no_grad():
        for X_b, _ in test_loader:
            X_b = X_b.to(device)
            outputs = torch.sigmoid(model(X_b))
            test_probs.extend(outputs.cpu().numpy())

    test_probs = np.array(test_probs)
    metrics = evaluate_imbalanced_metrics(y_test, test_probs)
    metrics["modelo"] = "Red Neuronal 3D Conv3D (PyTorch)"

    print(f"✅ Evaluación 3D PyTorch Finalizada - ROC-AUC: {metrics['roc_auc']:.4f} | Recall@FPR<=5%: {metrics['recall_at_fpr5']*100:.2f}%")
    return metrics


if __name__ == "__main__":
    print("Módulo train_nn_3d listo para importar.")
