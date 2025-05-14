
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('your_data_path.csv')  # Replace with actual path
data = data.sort_values(by=["Process_ID", 'OperationTime']).reset_index(drop=True)

input_cols = ['RetentionTime', 'OperationTime', 'Q_s', 'Q_F', 'Q_d']
output_col = "MethaneProduction"

# Normalize data
scaler_x = StandardScaler()
scaler_y = StandardScaler()
data[input_cols] = scaler_x.fit_transform(data[input_cols])
data[output_col] = scaler_y.fit_transform(data[[output_col]])

# Create sequences
seq_length = 17
sequences, targets = [], []
Q_s_list, Q_F_list, Q_d_list = [], [], []

for _, group in data.groupby("Process_ID"):
    x = group[input_cols].values
    y = group[output_col].values
    Q_s = group["Q_s"].values
    Q_F = group["Q_F"].values
    Q_d = group["Q_d"].values

    for i in range(len(group) - seq_length):
        sequences.append(x[i:i + seq_length])
        targets.append(y[i + seq_length])
        Q_s_list.append(Q_s[i + seq_length])
        Q_F_list.append(Q_F[i + seq_length])
        Q_d_list.append(Q_d[i + seq_length])

# Convert to tensors
X = torch.tensor(np.array(sequences), dtype=torch.float32)
y = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1)
Q_s = torch.tensor(np.array(Q_s_list), dtype=torch.float32).unsqueeze(1)
Q_F = torch.tensor(np.array(Q_F_list), dtype=torch.float32).unsqueeze(1)
Q_d = torch.tensor(np.array(Q_d_list), dtype=torch.float32).unsqueeze(1)

# Split into train and test datasets
dataset = TensorDataset(X, y, Q_s, Q_F, Q_d)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Define PI-LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.alpha = nn.Parameter(torch.ones(seq_length))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Define custom loss function
class PhysicsLoss(nn.Module):
    def __init__(self, model, lambda_phys=0.1):
        super().__init__()
        self.model = model
        self.lambda_phys = lambda_phys
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true, Q_s, Q_F, Q_d):
        mse_loss = self.mse(y_pred, y_true)
        converted_VS = Q_s + Q_F - Q_d
        constraint_loss = torch.mean((y_pred - self.model.alpha * converted_VS) ** 2)
        return mse_loss + self.lambda_phys * constraint_loss

# Initialize model
model = LSTMModel(input_size=len(input_cols), hidden_size=64, num_layers=2, output_size=1)
loss_fn = PhysicsLoss(model)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 100
train_losses, test_losses = [], []

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for X_batch, y_batch, Qs, QF, Qd in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch, Qs, QF, Qd)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    train_losses.append(total_train_loss / len(train_loader))

    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch, Qs, QF, Qd in test_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch, Qs, QF, Qd)
            total_test_loss += loss.item()
    test_losses.append(total_test_loss / len(test_loader))

    print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Test Loss = {test_losses[-1]:.4f}")

# Plot losses
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Test Loss")
plt.show()
