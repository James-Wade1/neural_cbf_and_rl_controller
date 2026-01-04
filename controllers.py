from abc import ABC, abstractmethod
import numpy as np
import torch
from ppo_controller import ActorCritic

class ControllerAbstract(ABC):
    @abstractmethod
    def get_control(self, robot_position: np.ndarray, goal_position: np.ndarray) -> np.ndarray:
        pass

class NominalController(ControllerAbstract):
    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def get_control(self, robot_position: np.ndarray, goal_position: np.ndarray) -> np.ndarray:
        u = goal_position - robot_position
        if np.linalg.norm(u) > 1e-6:
            u = u / (np.linalg.norm(u) + 1e-9)
        u = self.gain * u
        return u

class RLController(ControllerAbstract):
    def __init__(self, model: ActorCritic, device, gain: float = 1.0):
        self.model: ActorCritic = model
        self.device = device
        self.gain = gain

    def get_control(self, robot_position: np.ndarray, goal_position: np.ndarray) -> np.ndarray:
        rel_distance = goal_position - robot_position
        state = np.concatenate([robot_position, goal_position, rel_distance, np.array([0])]).astype(np.float32)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_tensor, _, _ =self.model.get_action(state_tensor, deterministic=True)
        action = action_tensor.cpu().numpy().squeeze(0)
        return self.gain * action