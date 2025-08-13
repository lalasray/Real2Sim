import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loader.utd_mhad import UTDMHAD_IMUDataset, collate_fn_no_pad
import numpy as np

# ----------------------
# DeepConvLSTM Model
# ----------------------
class DeepConvLSTM(nn.Module):
    def __init__(self, input_channels=6, n_classes=27, hidden_size=128, num_layers=2):
        super(DeepConvLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)
    
    def forward(self, x):
        # x: (B, C, W)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.transpose(1, 2)  # (B, W, 128)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # last time step
        out = self.fc(out)
        return out

# ----------------------
# Training / Evaluation
# ----------------------
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        acc = torch.stack([s['acc_sim'] for s in batch]).to(device)  # (B, W, 3)
        gyro = torch.stack([s['gyro_sim'] for s in batch]).to(device)  # (B, W, 3)
        x = torch.cat([acc, gyro], dim=-1).transpose(1, 2)  # (B, 6, W)

        # Convert to zero-based class indices
        labels = torch.tensor([s['activity'] for s in batch], dtype=torch.long).to(device) - 1

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            acc = torch.stack([s['acc_sim'] for s in batch]).to(device)
            gyro = torch.stack([s['gyro_sim'] for s in batch]).to(device)
            x = torch.cat([acc, gyro], dim=-1).transpose(1, 2)
            labels = torch.tensor([s['activity'] for s in batch], dtype=torch.long).to(device) - 1

            outputs = model(x)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

# ----------------------
# Main
# ----------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root_sim = '/home/lala/Documents/Data/VQIMU/UTD_MHAD'
    window_size = 24
    stride = 1

    # Dataset split: subjects 1-5 train, 6 val, 7-8 test
    train_dataset = UTDMHAD_IMUDataset(root_sim, subjects=[1, 2, 3, 4, 5], window_size=window_size, stride=stride)
    val_dataset = UTDMHAD_IMUDataset(root_sim, subjects=[6], window_size=window_size, stride=stride)
    test_dataset = UTDMHAD_IMUDataset(root_sim, subjects=[7, 8], window_size=window_size, stride=stride)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_no_pad)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_no_pad)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_no_pad)

    # Count unique activities in training set
    n_classes = len(set([activity for (_, _, activity) in train_dataset.samples]))
    print(f"Detected {n_classes} activity classes.")

    model = DeepConvLSTM(input_channels=6, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.4f}")

if __name__ == '__main__':
    main()
