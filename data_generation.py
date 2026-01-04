import pandas as pd
import numpy as np
from obstacles import Obstacle
from environment import Robot

def generate_cbf_training_data(num_samples: int,
                               num_scenarios: int = 50, max_obstacles: int = 4,
                               workspace_bounds: tuple[tuple[float, float], tuple[float, float]] = ((-1, 6), (-1, 7)),
                               frac_boundary: float = 0.4, frac_inside: float = 0.2, boundary_jitter: float = 0.03) -> pd.DataFrame:

    data = {
        'x': [], 'y': [],
        'h_true': [], 'grad_h_x': [], 'grad_h_y': [],
        'second_h_true': [], 'second_grad_h_x': [], 'second_grad_h_y': [],
        'u_nom_x': [], 'u_nom_y': [],
        'goal_x': [], 'goal_y': [],
        'num_obstacles': [], 'min_obstacle_radii': [], 'min_obstacle_center_x': [], 'min_obstacle_center_y': [],
        'min_obstacle_relative_x': [], 'min_obstacle_relative_y': [],
        'min_obstacle_distance': [],
        'second_obstacle_radii': [], 'second_obstacle_center_x': [], 'second_obstacle_center_y': [],
        'second_obstacle_relative_x': [], 'second_obstacle_relative_y': [],
        'second_obstacle_distance': []
    }

    samples_per_scenario = num_samples // num_scenarios
    rng = np.random.default_rng(42)

    (x_min, x_max), (y_min, y_max) = workspace_bounds

    def sample_obstacles(k) -> list[Obstacle]:
        obs_list = []
        for _ in range(k):
            cx = rng.uniform(x_min + 0.5, x_max - 0.5)
            cy = rng.uniform(y_min + 0.5, y_max - 0.5)
            radius = rng.uniform(0.2, 0.8)
            obs_list.append(Obstacle(radius=radius, center = np.array([cx, cy])))
        return obs_list

    def calc_h_and_grad_and_store_in_data(robot: Robot, obstacles: list[Obstacle], goal_pos: np.ndarray):
        u_nom = robot.get_nominal_ctrl(goal_pos)

        h_vals = [obs.h(robot.get_position()) for obs in obstacles]
        h_grad_vals = [obs.h_grad(robot.get_position()) for obs in obstacles]
        indices = np.argsort(h_vals)
        idx = indices[0]
        idx2 = indices[1] if len(indices) > 1 else idx
        h_min = h_vals[idx]
        h_grad_min = h_grad_vals[idx]
        obs_min = obstacles[idx]

        h_second_min = h_vals[idx2]
        h_grad_second_min = h_grad_vals[idx2]
        obs_second_min = obstacles[idx2]

        data['x'].append(robot.position[0])
        data['y'].append(robot.position[1])
        data['h_true'].append(h_min)
        data['grad_h_x'].append(h_grad_min[0])
        data['grad_h_y'].append(h_grad_min[1])
        data['second_h_true'].append(h_second_min)
        data['second_grad_h_x'].append(h_grad_second_min[0])
        data['second_grad_h_y'].append(h_grad_second_min[1])
        data['u_nom_x'].append(u_nom[0])
        data['u_nom_y'].append(u_nom[1])
        data['goal_x'].append(goal_pos[0])
        data['goal_y'].append(goal_pos[1])
        data['num_obstacles'].append(len(obstacles))
        data['min_obstacle_radii'].append(obs_min.radius)
        data['min_obstacle_center_x'].append(obs_min.center[0])
        data['min_obstacle_center_y'].append(obs_min.center[1])
        data['min_obstacle_relative_x'].append(obs_min.center[0] - robot.position[0])
        data['min_obstacle_relative_y'].append(obs_min.center[1] - robot.position[1])
        data['min_obstacle_distance'].append(np.linalg.norm(obs_min.center - robot.position) - obs_min.radius)
        data['second_obstacle_radii'].append(obs_second_min.radius)
        data['second_obstacle_center_x'].append(obs_second_min.center[0])
        data['second_obstacle_center_y'].append(obs_second_min.center[1])
        data['second_obstacle_relative_x'].append(obs_second_min.center[0] - robot.position[0])
        data['second_obstacle_relative_y'].append(obs_second_min.center[1] - robot.position[1])
        data['second_obstacle_distance'].append(np.linalg.norm(obs_second_min.center - robot.position) - obs_second_min.radius)

    for s in range(num_scenarios):
        print(f"Generating scenario {s + 1}")
        num_obstacles = rng.integers(1, max_obstacles + 1)
        obstacles = sample_obstacles(num_obstacles)

        goal_pos = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)])

        n_random = int((1-frac_boundary-frac_inside)*samples_per_scenario)
        n_boundary = int(frac_boundary*samples_per_scenario)
        n_inside = int(frac_inside*samples_per_scenario)

        for _ in range(n_random):
            robot = Robot(radius=0.05, position=np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)]))
            calc_h_and_grad_and_store_in_data(robot, obstacles, goal_pos)

        for _ in range(n_boundary):
            obs: Obstacle = rng.choice(obstacles)
            theta = rng.uniform(0, 2*np.pi)
            r = obs.radius + rng.normal(0, boundary_jitter*obs.radius)
            pos = obs.center + np.array([r*np.cos(theta),r*np.sin(theta)])
            robot = Robot(radius=0.05, position=pos)
            calc_h_and_grad_and_store_in_data(robot, obstacles, goal_pos)

        for _ in range(n_inside):
            obs: Obstacle = rng.choice(obstacles)
            theta = rng.uniform(0, 2*np.pi)
            r = rng.uniform(0, obs.radius*0.9)
            pos = obs.center + np.array([r*np.cos(theta),r*np.sin(theta)])
            robot = Robot(radius=0.05, position=pos)
            calc_h_and_grad_and_store_in_data(robot, obstacles, goal_pos)

    return pd.DataFrame(data)

if __name__ == "__main__":
    data = generate_cbf_training_data(num_samples=10000)
    data.to_csv("final_project_data/cbf_training_data.csv", index=False)