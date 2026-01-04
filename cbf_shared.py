import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

class CBFDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame):
        self.data_frame = data_frame

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data_frame.iloc[idx]
        input_features = torch.tensor([
            row['x'], row['y'],
            row['min_obstacle_radii'], row['min_obstacle_center_x'], row['min_obstacle_center_y'],
            row['min_obstacle_distance'],
        ], dtype=torch.float32)
        h_true = torch.tensor([row['h_true'], row['grad_h_x'], row['grad_h_y']], dtype=torch.float32)
        return input_features, h_true

class CBFLoss(nn.Module):
    def __init__(self, lambda_grad: float = 2.5):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_grad = lambda_grad

    def forward(self, h_pred: torch.Tensor, grad_h_pred: torch.Tensor, h_true: torch.Tensor, grad_h_true: torch.Tensor) -> tuple[torch.Tensor, dict]:
        loss_h = self.mse_loss(h_pred, h_true)

        loss_grad = self.mse_loss(grad_h_pred, grad_h_true)

        total_loss = (
            loss_h +
            self.lambda_grad * loss_grad
        )

        return total_loss, {
            'h': loss_h.item(),
            'grad': loss_grad.item()
        }