import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Load and preprocess data
data_dir = (
    "C:/Users/jed95/Documents/GitHub/anomaly_detection/dataset/yahoo_s5/A2Benchmark"
)
file_list = [
    os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")
]
data_frames = []
for file in file_list:
    df = pd.read_csv(file)
    data_frames.append(df)
data = pd.concat(data_frames, ignore_index=True)

# Preprocess data
values = data["value"].values.reshape(-1, 1)
scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)
window_size = 10
train_ratio = 0.51


def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size):
        window = data[i : i + window_size]
        windows.append(window)
    return np.array(windows)


windows = create_windows(values_scaled, window_size)
train_size = int(len(windows) * train_ratio)
train_data = windows[:train_size]
test_data = windows[train_size:]


class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].transpose(1, 0)


train_dataset = TimeSeriesDataset(train_data)
test_dataset = TimeSeriesDataset(test_data)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

from source.utilsVAEs5 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 10
num_param = 1
scale_flag = 0
model = VAE(latent_dim, num_param, window_size, scale_flag).to(device)


# 3. Define the training function
def likelihood_loss(x_recon, x, metric="MSE"):
    if metric == "MSE":
        loss = nn.MSELoss(reduction="none")
    elif metric == "BCE":
        loss = nn.BCELoss(reduction="none")
    else:
        raise ValueError("Invalid metric. Use 'MSE' or 'BCE'.")
    return loss(x_recon, x).mean(dim=[1, 2])


def train_model(model, optimizer, train_loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        rec_loss_total = 0
        kl_loss_total = 0

        for X_batch in train_loader:
            X_batch = X_batch.to(device)

            optimizer.zero_grad()
            x_rec = model(X_batch)

            likelihood = likelihood_loss(x_rec, X_batch, metric="MSE")
            rec_loss = torch.mean(likelihood)
            kl_div = torch.mean(model.kl_div)
            loss = rec_loss + kl_div * 0.1

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            rec_loss_total += rec_loss.item()
            kl_loss_total += kl_div.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Total Loss: {total_loss:.4f}, "
            f"Rec Loss: {rec_loss_total:.4f}, "
            f"KL Div: {kl_loss_total:.4f}"
        )
    return model


# 4. Train the model
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 10
model = train_model(model, optimizer, train_loader, num_epochs=num_epochs)

# 5. Anomaly detection
model.eval()
reconstruction_errors = []
all_true_labels = []

with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        x_rec = model(X_batch)
        error = likelihood_loss(x_rec, X_batch, metric="MSE").cpu().numpy()
        reconstruction_errors.append(error)

        if "is_anomaly" in data.columns:
            idx = len(reconstruction_errors) - 1
            true_label = data["is_anomaly"].values[train_size + idx + window_size]
            all_true_labels.append(true_label)

reconstruction_errors = np.concatenate(reconstruction_errors)

# Set anomaly threshold
k = 2.5
threshold = reconstruction_errors.mean() + k * reconstruction_errors.std()
print(f"Anomaly Threshold: {threshold:.4f}")

anomalies = reconstruction_errors > threshold

# Evaluate performance
if "is_anomaly" in data.columns:
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    true_labels = np.array(all_true_labels)
    pred_labels = anomalies.astype(int)

    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

# Visualize results
plt.figure(figsize=(15, 5))
plt.plot(reconstruction_errors, label="Reconstruction Error")
plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold")
plt.legend()
plt.show()

test_values = values_scaled[train_size + window_size :]

plt.figure(figsize=(15, 5))
plt.plot(test_values, label="Scaled Value")
anomaly_indices = np.where(anomalies)[0]
plt.scatter(anomaly_indices, test_values[anomaly_indices], color="r", label="Anomalies")
plt.legend()
plt.show()
