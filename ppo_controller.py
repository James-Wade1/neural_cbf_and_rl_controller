import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gymnasium as gym
from concurrent.futures import ThreadPoolExecutor, as_completed
from obstacles import Obstacle
from multiprocessing import cpu_count
from matplotlib import pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

class Robot2DEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.obstacles = []
        self.dt = 0.1
        self.max_steps = 1200
        self.current_step = 0
        self.trajectory = []
        self.current_state = None

        self._generate_obstacles(num_obstacles=3)

    def _generate_obstacles(self, num_obstacles: int):
        self.obstacles = []
        for _ in range(num_obstacles):
            center = self.np_random.uniform(low=0.0, high=6.0, size=(2,))
            radius = self.np_random.uniform(low=0.2, high=0.8)
            self.obstacles.append(Obstacle(center=center, radius=radius))

    def _get_obs(self):
        rel_goal = self.goal_pos - self.robot_pos
        return np.concatenate([self.robot_pos, self.goal_pos, rel_goal, [self._get_obs_distance()]])

    def _get_obs_distance(self):
        distances = [np.linalg.norm(self.robot_pos - obs.center) - obs.radius for obs in self.obstacles]
        return np.min(distances)

    def _check_done(self):
        return np.linalg.norm(self.robot_pos - self.goal_pos) < 0.01

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.current_step = 0

        self.robot_pos = self.np_random.uniform(low=0.0, high=6.0, size=(2,))
        self.goal_pos = self.np_random.uniform(low=0.0, high=6.0, size=(2,))
        self.trajectory = [self.robot_pos.copy()]
        self.current_state = self._get_obs()

        return self._get_obs(), {}

    def _compute_reward(self, action, prev_state):
        distance_to_goal = np.linalg.norm(self.current_state[4:6])
        prev_distance_to_goal = np.linalg.norm(prev_state[4:6])

        progress_reward = (prev_distance_to_goal - distance_to_goal)*20.0

        if self._check_done():
            goal_reward = 100.0
        else:
            goal_reward = 0.0

        obstacle_dist = self.current_state[6]
        danger_penalty = 0.0
        if obstacle_dist < 0.2 and obstacle_dist > 0.0:
            danger_penalty = -50.0 * (0.2 - obstacle_dist)
        if obstacle_dist <= 0.0:
            danger_penalty = -100.0

        time_penalty = -0.05

        # reward = progress_reward + goal_reward + danger_penalty + time_penalty
        reward = progress_reward + goal_reward
        return reward


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.robot_pos += action * self.dt
        self.current_step += 1
        self.trajectory.append(self.robot_pos.copy())
        old_state = self.current_state
        self.current_state = self._get_obs()

        done = self._check_done()

        truncated = self.current_step >= self.max_steps
        reward = self._compute_reward(action, old_state)

        return self.current_state, reward, done, truncated, {}

    def render(self, model, mode='human'):
        from environment import Visualizer, Environment, Robot
        from controllers import RLController
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        visualizer = Visualizer(Environment(Robot(radius=0.2, position=self.trajectory[-1], controller=RLController(model=model, device=device, gain=1.0)), self.obstacles, goal=self.goal_pos))
        visualizer.render(np.array(self.trajectory), anim=False, title="PPO Robot Trajectory")
        visualizer.plot_velocity_vector_field(goal=self.goal_pos, grid_resolution=20, title="PPO Robot Velocity Field")

class VectorizedRobot2DEnv(gym.vector.VectorEnv):
    def __init__(self, num_envs: int):
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        super().__init__()
        self.num_envs = num_envs
        self.envs = [Robot2DEnv() for _ in range(num_envs)]
        self.current_steps = np.zeros(num_envs, dtype=int)

    def reset(self, seed=None):
        observations = []
        for i, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + i
            obs, _ = env.reset(seed=env_seed)
            observations.append(obs)
        return np.array(observations), {}

    def step(self, actions):
        observations, rewards, dones, truncs, infos = [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, trunc, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncs.append(trunc)
            infos.append(info)

            self.current_steps[i] += 1

            if done or trunc:
                env.reset()
                self.current_steps[i] = 0
        return (np.array(observations), np.array(rewards), np.array(dones), np.array(truncs), infos)


class PPODataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ActorCritic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )

        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state: torch.Tensor):
        return self.shared_layers(state)

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        shared_out = self.forward(state)
        mean = self.actor_mean(shared_out)

        if deterministic:
            return mean, None, None
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        # action = torch.clamp(action, -1.0, 1.0)
        return action, log_prob, dist

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        shared_out = self.forward(state)
        mean = self.actor_mean(shared_out)
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(shared_out).squeeze(-1)

        return log_prob, value, entropy

class ResActorCritic(nn.Module):

    class ResidualBlock(nn.Module):
        def __init__(self, size: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.LayerNorm(size),
                nn.ReLU(),
                nn.Linear(size, size),
                nn.LayerNorm(size),
                nn.ReLU(),
                nn.Linear(size, size)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = x
            out = self.block(x)
            out += identity
            return out

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        shared_layers_list = nn.ModuleList()
        shared_layers_list.append(nn.Linear(state_dim, hidden_size))
        for _ in range(2):
            shared_layers_list.append(self.ResidualBlock(hidden_size))
        self.shared_layers = nn.Sequential(*shared_layers_list)

        actor_mean_list = nn.ModuleList()
        for _ in range(2):
            actor_mean_list.append(self.ResidualBlock(hidden_size))
        actor_mean_list.append(nn.Linear(hidden_size, action_dim))
        actor_mean_list.append(nn.Tanh())
        self.actor_mean = nn.Sequential(*actor_mean_list)

        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        critic_list = nn.ModuleList()
        for _ in range(2):
            critic_list.append(self.ResidualBlock(hidden_size))
        critic_list.append(nn.Linear(hidden_size, 1))

        self.critic = nn.Sequential(*critic_list)

    def forward(self, state: torch.Tensor):
        return self.shared_layers(state)

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        shared_out = self.forward(state)
        mean = self.actor_mean(shared_out)

        if deterministic:
            return mean, None, None
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, dist

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        shared_out = self.forward(state)
        mean = self.actor_mean(shared_out)
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(shared_out).squeeze(-1)

        return log_prob, value, entropy

class PPOController:
    def __init__(self, state_dim: int, action_dim: int, device: torch.device, lr: float = 0.001, gamma: float = 0.99, clip_epsilon: float = 0.2, gae_lambda: float = 0.95):
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda

        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action, log_prob, _ = self.model.get_action(state_tensor, deterministic)
        return action.cpu().numpy().squeeze(), log_prob.cpu().item() if log_prob is not None else None

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        return self.model.evaluate_actions(states, actions)

    def compute_gae(self, rewards, values, dones, next_value):
        values = values + [next_value]
        gae = 0
        n = len(rewards)
        advantages = [0]*n

        for step in reversed(range(n)):
            mask = 1.0 - float(dones[step])
            delta = rewards[step] + self.gamma*values[step + 1]*mask - values[step]
            gae = delta + self.gamma*self.gae_lambda*mask*gae

            advantages[step] = gae

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32).to(self.device)
        return advantages, returns

    def update(self, dataloader: DataLoader, epochs: int):
        for _ in range(epochs):
            for states, actions, old_log_probs, returns, advantages in dataloader:
                states = states.to(self.device).float()
                actions = actions.to(self.device).float()
                old_log_probs = old_log_probs.to(self.device).float()
                returns = returns.to(self.device).float()
                advantages = advantages.to(self.device).float()

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                log_probs, values, entropy = self.evaluate_actions(states, actions)

                ratios = torch.exp(log_probs - old_log_probs)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(values, returns)
                entropy_loss = -0.01 * entropy.mean()

                loss = actor_loss + critic_loss + entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

def collect_vectorized_rollout(controller: PPOController, vec_env: VectorizedRobot2DEnv, num_steps=2048, gamma=0.99, gae_lambda=0.95, device='cuda'):
    states_list = []
    actions_list = []
    log_probs_list = []
    rewards_list = []
    dones_list = []
    values_list = []

    # Track episode statistics
    episode_successes = 0
    episode_lengths = []
    current_episode_steps = np.zeros(vec_env.num_envs, dtype=int)

    states, _ = vec_env.reset()

    for step in range(num_steps):
        with torch.no_grad():
            states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
            actions, log_probs, _ = controller.model.get_action(states_tensor, deterministic=False)
            _, value, _ = controller.evaluate_actions(
                states_tensor,
                actions.to(device)
            )
            actions = actions.cpu().numpy()
            log_probs = log_probs.cpu().numpy()
            value = value.cpu().numpy()

        next_states, rewards, dones, truncs, _ = vec_env.step(actions)

        # Track episode statistics before environments auto-reset
        current_episode_steps += 1
        for i in range(vec_env.num_envs):
            if dones[i] or truncs[i]:
                episode_lengths.append(current_episode_steps[i])
                if dones[i]:  # done=True means goal reached, truncated=True means timeout
                    episode_successes += 1
                current_episode_steps[i] = 0

        states_list.append(states)
        actions_list.append(actions)
        log_probs_list.append(log_probs)
        rewards_list.append(rewards)
        dones_list.append(dones | truncs)
        values_list.append(value)

        states = next_states

    with torch.no_grad():
        final_states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        _, next_value, _ = controller.evaluate_actions(
            final_states_tensor,
            torch.tensor(actions, dtype=torch.float32).to(device)
        )
        next_value = next_value.cpu().numpy()

    states_array = np.array(states_list)
    actions_array = np.array(actions_list)
    log_probs_array = np.array(log_probs_list)
    rewards_array = np.array(rewards_list)
    dones_array = np.array(dones_list)
    values_array = np.array(values_list)

    advantages_array = np.zeros_like(rewards_array)
    last_gae = np.zeros(vec_env.num_envs)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_values = next_value
        else:
            next_values = values_array[t + 1]

        mask = 1.0 - dones_array[t]
        deltas = rewards_array[t] + gamma*next_values*mask - values_array[t]
        last_gae = deltas + gamma*gae_lambda*mask*last_gae
        advantages_array[t] = last_gae

    returns_array = advantages_array + values_array

    states_array = states_array.reshape(-1, states_array.shape[-1])
    actions_array = actions_array.reshape(-1, actions_array.shape[-1])
    log_probs_array = log_probs_array.reshape(-1)
    returns_array = returns_array.reshape(-1)
    advantages_array = advantages_array.reshape(-1)
    values_array = values_array.reshape(-1)

    return {
        "states": states_array,
        "actions": actions_array,
        "log_probs": log_probs_array,
        "returns": returns_array,
        "advantages": advantages_array,
        "values": values_array,
        "episode_successes": episode_successes,
        "episode_lengths": episode_lengths,
        "total_episodes": len(episode_lengths)
    }

def train_ppo(num_epochs: int = 20, num_steps_per_epoch: int = 2048, render: bool = True, num_envs: int = 8, render_interval: int = 1, batch_size: int = 64) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # Orthogonal init keeps gradients clean in deep networks
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
    controller = PPOController(state_dim=7, action_dim=2, device=device)
    controller.model.apply(init_weights)
    vec_env = VectorizedRobot2DEnv(num_envs=num_envs)

    # Track metrics
    epoch_mean_returns = []
    epoch_mean_rewards = []
    epoch_mean_lengths = []
    epoch_success_rates = []

    for epoch in tqdm(range(num_epochs), desc="PPO Training Epochs"):
        if render and epoch % render_interval == 0:
            with torch.no_grad():
                env = vec_env.envs[0]
                state, _ = env.reset()
                done = False
                truncated = False
                while not done and not truncated:
                    action, _ = controller.select_action(state, deterministic=True)
                    state, _, done, truncated, _ = env.step(action)
                env.render(model=controller.model)
                plt.pause(0.1)
                plt.close('all')
        data = collect_vectorized_rollout(controller, vec_env, device=device, num_steps=num_steps_per_epoch)

        # Record metrics for this epoch using statistics from rollout
        epoch_mean_returns.append(float(data["returns"].mean()))
        epoch_mean_rewards.append(float(data["returns"].mean() / (num_steps_per_epoch / num_envs + 1e-8)))

        if data["episode_lengths"]:
            epoch_mean_lengths.append(float(np.mean(data["episode_lengths"])))
        else:
            epoch_mean_lengths.append(0.0)

        if data["total_episodes"] > 0:
            epoch_success_rates.append(float(data["episode_successes"] / data["total_episodes"]))
        else:
            epoch_success_rates.append(0.0)

        dataset = PPODataset(list(zip(
            data["states"],
            data["actions"],
            data["log_probs"],
            data["returns"],
            data["advantages"])))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        controller.update(dataloader, epochs=10)

    with torch.no_grad():
        env = vec_env.envs[0]
        state, _ = env.reset()
        done = False
        truncated = False
        while not done and not truncated:
            action, _ = controller.select_action(state, deterministic=True)
            state, _, done, truncated, _ = env.step(action)
        env.render(model=controller.model)
        plt.pause(0.1)
        plt.close('all')

    # Plot training metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Mean return per epoch
    axes[0, 0].plot(epoch_mean_returns, linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Mean Return', fontsize=12)
    axes[0, 0].set_title('Mean Episode Return over Training', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Mean episode length per epoch
    axes[0, 1].plot(epoch_mean_lengths, linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Mean Episode Length (steps)', fontsize=12)
    axes[0, 1].set_title('Mean Episode Length over Training', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Success rate per epoch
    axes[1, 0].plot(epoch_success_rates, linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Success Rate', fontsize=12)
    axes[1, 0].set_title('Success Rate over Training', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.05])

    # Plot 4: Efficiency (reward per step)
    if epoch_mean_lengths and len(epoch_mean_returns) == len(epoch_mean_lengths):
        efficiency = np.array(epoch_mean_returns) / (np.array(epoch_mean_lengths) + 1e-8)
        axes[1, 1].plot(efficiency, linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Return per Step', fontsize=12)
        axes[1, 1].set_title('Training Efficiency (Return/Step)', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('final_project_data/ppo_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

    # torch.save(controller.model, "final_project_data/ppo_controller_model2.pth")

if __name__ == "__main__":
    train_ppo(num_epochs=20, num_steps_per_epoch=2048, render=False, num_envs=8, render_interval=10, batch_size=128)