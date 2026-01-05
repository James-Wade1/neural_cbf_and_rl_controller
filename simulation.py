import torch
from environment import Environment, Robot, Visualizer
from obstacles import Obstacle, BarrierFunctionNeuralNetwork
from trajectory_optimizer import TrajectoryOptimizer
from fully_connected_nn import FullyConnectedCBF
from resnet import ResNetCBF, ResidualBlock
from ppo_controller import ActorCritic
import numpy as np
from controllers import NominalController, RLController

np.random.seed(42)
torch.manual_seed(42)

radii = np.random.uniform(0.2, 0.8, 5)

def run(model_path: str = "", title: str = "", goal_idx: int = 0, use_rl: bool = False, save_animation: str = None) -> None:
    obstacles = [
        Obstacle(radius=radii[0], center=np.array([1.5, 1.0]), color="red"),
        Obstacle(radius=radii[1], center=np.array([3.5, 3.0]), color="green"),
        Obstacle(radius=radii[2], center=np.array([4.0, 1.0]), color="purple"),
        Obstacle(radius=radii[3], center=np.array([2.0, 4.0]), color="blue"),
        Obstacle(radius=0.6, center=np.array([1.0, 3.0]), color="orange"),
        Obstacle(radius=radii[4], center=np.array([3.5, 5.0]), color="cyan"),
    ]

    goal_poses = [np.array([5.0, 5.0]), np.array([2.0, 5.0]), np.array([5.0, 3.0])]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model = model.to(device).eval()
        for o in obstacles:
            o.barrier_function = BarrierFunctionNeuralNetwork(model, device)
    robot = Robot(radius=0.2, position=np.array([0.0, 0.0]))

    if use_rl:
        model: ActorCritic = torch.load("final_project_data/ppo_controller_model.pth", map_location=device, weights_only=False)
        model = model.to(device).eval()
        controller = RLController(model=model, device=device, gain=1.0)
        robot.controller = controller

    goal_pos = goal_poses[goal_idx]
    env = Environment(robot=robot, obstacles=obstacles)
    optimizer = TrajectoryOptimizer(env=env, total_time=1.0, num_steps=100)
    trajectory = optimizer.run(max_iters=2000, goal=goal_pos)
    print("Optimized Trajectory:")
    print(trajectory)

    visualizer = Visualizer(env=env)
    visualizer.render(trajectory, anim=True, title=title, save_path=save_animation)
    visualizer.plot_min_h_values(grid_resolution=100, title=title)
    visualizer.plot_velocity_vector_field(goal=goal_pos, grid_resolution=20, title="PPO Controller")

if __name__ == "__main__":
    run(model_path="final_project_data/fully_connected_cbf_model.pth", title="Fully Connected CBF Model and RL Controller", goal_idx=1, use_rl=True)
    run(model_path="final_project_data/resnet_cbf_model.pth", title="ResNet CBF Model and RL Controller", goal_idx=1, use_rl=True, save_animation="resnet_cbf_rl_animation.mp4")
    run(title="Analytical CBFs and Analytical Controller", goal_idx=1, use_rl=False, save_animation="analytical_cbf_animation.mp4")