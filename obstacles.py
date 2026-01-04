from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import torch

@dataclass
class BarrierFunctionArgs:
    x: np.ndarray
    obs_x: float
    obs_y: float
    obs_r: float
    distance: float

class BarrierFunctionAbstract(ABC):
    @abstractmethod
    def h(self, x: np.ndarray, distance: float) -> float:
        pass
    @abstractmethod
    def h_grad(self, x: np.ndarray, args: BarrierFunctionArgs) -> np.ndarray:
        pass

class BarrierFunctionAnalytical(BarrierFunctionAbstract):
    def __init__(self, radius: float, center: np.ndarray, alpha: float = 1.0):
        super().__init__()
        self.radius: float = radius
        self.center: np.ndarray = center
        self.alpha: float = alpha  # Barrier function parameter

    def h(self, x: np.ndarray, args: BarrierFunctionArgs = None) -> float:
        """Evaluate the barrier function at point x."""
        return np.linalg.norm(x - self.center)**2 - self.radius**2

    def h_grad(self, x: np.ndarray, args: BarrierFunctionArgs = None) -> np.ndarray:
        """Evaluate the gradient of the barrier function at point x."""
        return 2 * (x - self.center)

class BarrierFunctionNeuralNetwork(BarrierFunctionAbstract):
    def __init__(self, model: torch.nn.Module, device: torch.device, alpha: float = 1.0):
        super().__init__()
        self.model: torch.nn.Module = model
        self.device: torch.device = device
        self.alpha: float = alpha

    def generate_input_vector(self, x: np.ndarray, args: BarrierFunctionArgs) -> torch.Tensor:
        """Generate the input vector for the neural network based on position and obstacle info."""
        input_vector = torch.tensor([x[0], x[1], args.obs_r, args.obs_x, args.obs_y, args.distance], dtype=torch.float32).to(self.device).unsqueeze(0)
        return input_vector

    def h(self, x: np.ndarray, args: BarrierFunctionArgs = None) -> float:
        """Evaluate the barrier function at point x using the neural network."""
        x_tensor = self.generate_input_vector(x, args)
        with torch.no_grad():
            h_value = self.model(x_tensor).item()
        return h_value

    def h_grad(self, x: np.ndarray, args: BarrierFunctionArgs = None) -> np.ndarray:
        """Evaluate the gradient of the barrier function at point x using autograd."""
        x_tensor = self.generate_input_vector(x, args)
        x_tensor.requires_grad_(True)
        grad = self.model.compute_grad_h(x_tensor)
        return grad.detach().cpu().numpy().squeeze()

class Obstacle:
    def __init__(self, radius: float, center: np.ndarray, color: str = "red", barrier_function: BarrierFunctionAbstract = None):
        self.center: np.ndarray = center
        self.radius: float = radius
        self.barrier_function = barrier_function if barrier_function else BarrierFunctionAnalytical(radius, center, 2.0)
        self.color: str = color
        self.args = BarrierFunctionArgs(
            x=None,
            obs_x=center[0],
            obs_y=center[1],
            obs_r=radius,
            distance=None
        )

    def h(self, x: np.ndarray) -> float:
        self.args.x = x
        self.args.distance = np.linalg.norm(x - self.center) - self.radius
        return self.barrier_function.h(x, self.args)

    def h_grad(self, x: np.ndarray) -> np.ndarray:
        self.args.x = x
        self.args.distance = np.linalg.norm(x - self.center) - self.radius
        return self.barrier_function.h_grad(x, self.args)