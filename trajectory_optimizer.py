import numpy as np
import cvxpy as cp
from environment import Environment

class TrajectoryOptimizer:
    def __init__(self, env: Environment, total_time: float, num_steps: int):
        self.env: Environment = env
        self.total_time: float = total_time
        self.num_steps: int = num_steps
        self.dt: float = total_time / num_steps

    def step(self, u):
        self.env.step(u, self.dt)

    def run(self, max_iters: int = 1000, goal: np.ndarray = np.array([5.0, 5.0])) -> np.ndarray:
        num_step = 0
        traj = [self.env.robot.position.copy()]
        self.env.set_goal(goal)
        while self.env.get_positional_error() > 0.01 and num_step < max_iters:
            u_nom = self.env.get_robot_nominal_ctrl(goal)

            u = cp.Variable(u_nom.shape)
            obj = 0.5 * cp.sum_squares(u - u_nom)  # Placeholder objective
            constraints = []
            for obs in self.env.obstacles:
                h_val = obs.h(self.env.robot.position)
                h_grad = obs.h_grad(self.env.robot.position)
                constraint = h_grad.T @ u + obs.barrier_function.alpha * h_val >= 0
                constraints.append(constraint)

            prob = cp.Problem(cp.Minimize(obj), constraints)
            try:
                prob.solve()
            except Exception as e:
                print(f"Optimization failed at step {num_step}: {e}")
                break

            u_opt = np.array(u.value).reshape(u_nom.shape)
            self.step(u_opt)
            num_step += 1
            traj.append(self.env.robot.position.copy())
        return np.array(traj)