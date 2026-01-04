from cbf_shared import CBFDataset, CBFLoss
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

class ResidualBlock(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(size),
            nn.SiLU(),
            nn.Linear(size, size),
            nn.LayerNorm(size),
            nn.SiLU(),
            nn.Linear(size, size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.block(x)
        out += identity
        return out

class ResNetCBF(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 6):
        super().__init__()
        layers = nn.ModuleList()
        layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 1):  # Two hidden layers
            layers.append(ResidualBlock(hidden_size))
        layers.append(nn.Linear(hidden_size, 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def compute_grad_h(self, x: torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        h = self.forward(x)
        grad_h = torch.autograd.grad(outputs=h, inputs=x,
                                     grad_outputs=torch.ones_like(h),
                                     create_graph=True)[0]
        return grad_h[:, :2]  # Return only gradient w.r.t. position (first two dimensions)

def validation_loss(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    for inputs, h_and_grad_true in val_loader:
        inputs = inputs.to(device)
        h_and_grad_true = h_and_grad_true.to(device)
        h_true, grad_h_true = h_and_grad_true[:, 0], h_and_grad_true[:, 1:3]
        h_pred = model(inputs)
        grad_h_pred = model.compute_grad_h(inputs)

        loss, _ = criterion(h_pred, grad_h_pred, h_true.unsqueeze(1), grad_h_true)
        val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    model.train()
    return avg_val_loss

def train_neural_cbf(model, train_loader, val_loader, optimizer, criterion, device, num_epochs: int = 100):
    model.train()
    train_loss = []
    val_loss = []
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        epoch_loss = 0.0
        for inputs, h_and_grad_true in train_loader:
            inputs = inputs.to(device)
            h_and_grad_true = h_and_grad_true.to(device)
            h_true, grad_h_true = h_and_grad_true[:, 0], h_and_grad_true[:, 1:3]
            optimizer.zero_grad()
            h_pred = model(inputs)
            grad_h_pred = model.compute_grad_h(inputs)

            loss, loss_dict = criterion(h_pred, grad_h_pred, h_true.unsqueeze(1), grad_h_true)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_epoch_loss)
        val_epoch_loss = validation_loss(model, val_loader, criterion, device)
        val_loss.append(val_epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Details: {loss_dict}")

    return train_loss, val_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ResNetCBF(input_size=6, hidden_size=128, num_layers=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = CBFLoss(lambda_grad=2.5).to(device)

    data_frame = pd.read_csv("final_project_data/cbf_training_data.csv")
    full_dataset = CBFDataset(data_frame)
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    train_loss, val_loss = train_neural_cbf(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=50)

    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss for ResNet CBF Model')
    plt.legend()
    plt.show()

    model.eval()
    torch.save(model, "final_project_data/resnet_cbf_model.pth")