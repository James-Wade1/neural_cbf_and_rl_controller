# Control Barrier Functions with Deep Learning and Reinforcement Learning

This project explores two machine learning approaches for robot navigation in a 2D environment with circular obstacles:

1. **Learning Control Barrier Functions (CBFs)** via supervised learning from analytical barrier function data
2. **RL-based controller** using Proximal Policy Optimization (PPO) to imitate a straight-line navigation controller

## Project Overview

### 1. Control Barrier Functions (CBFs)

The project learns neural network approximations of Control Barrier Functions for a 2D single integrator system navigating around circular obstacles. The CBF ensures safe navigation by constraining control inputs to maintain positive values of a barrier function.

**Key Features:**
- Analytical barrier function based on minimum distance to obstacles
- Data generation with strategic sampling (boundary, interior, and exterior points)
- Neural network architectures: Fully Connected and ResNet
- Loss function combines barrier value and gradient matching
- Integration with trajectory optimization for safe path planning

### 2. PPO-based Controller

An RL-based controller trained using Proximal Policy Optimization to imitate a simple straight-line controller (proportional control from current position to goal).

**Key Features:**
- Custom Gym environment for 2D robot navigation
- Actor-Critic architecture with separate policy and value networks
- Training uses demonstrations from nominal straight-line controller
- Obstacle-aware state representation including nearest obstacle distance

## Project Structure

```
.
├── cbf_shared.py              # Shared CBF dataset and loss function
├── controllers.py             # Controller implementations (nominal, RL, CBF-QP)
├── data_generation.py         # Generate training data from analytical CBF
├── environment.py             # 2D robot environment and visualization
├── fully_connected_nn.py      # Fully connected neural network CBF model
├── resnet.py                  # ResNet-based CBF model
├── ppo_controller.py          # PPO training and Actor-Critic architecture
├── obstacles.py               # Obstacle class and analytical barrier functions
├── trajectory_optimizer.py    # CBF-constrained trajectory optimization
└── simulation.py              # Run simulations with trained models
```

## Core Components

### CBF Learning Pipeline

1. **Data Generation** (`data_generation.py`):
   - Samples robot positions near obstacle boundaries, inside obstacles, and in free space
   - Computes analytical barrier function values and gradients
   - Generates diverse scenarios with varying numbers of obstacles

2. **Neural Network Models**:
   - **Fully Connected** (`fully_connected_nn.py`): Multi-layer perceptron with SiLU activations
   - **ResNet** (`resnet.py`): Residual network for learning complex barrier functions

3. **Loss Function** (`cbf_shared.py`):
   - Combines MSE loss on barrier values and gradients
   - Weighted combination: `L = L_h + λ_grad * L_grad`

4. **Trajectory Optimization** (`trajectory_optimizer.py`):
   - Uses learned CBF in QP-based control
   - Ensures safety constraints while reaching goal

### PPO Controller Pipeline

1. **Environment** (`ppo_controller.py`):
   - State: [robot_pos (2), goal_pos (2), relative_goal (2), min_obstacle_distance (1)]
   - Action: 2D velocity command
   - Reward: Progress toward goal with penalties for collisions

2. **Actor-Critic Architecture** (`ppo_controller.py`):
   - Shared feature extractor
   - Separate policy (actor) and value (critic) heads
   - Gaussian policy for continuous action space

3. **Training** (`ppo_controller.py`):
   - Uses PPO algorithm with clipped objective
   - Trains on episodes collected from environment
   - Imitation learning from nominal controller demonstrations

## Usage

### Training CBF Models

```python
# Generate training data
from data_generation import generate_cbf_training_data
data = generate_cbf_training_data(num_samples=50000, num_scenarios=50)

# Train fully connected model
from fully_connected_nn import FullyConnectedCBF
model = FullyConnectedCBF(input_size=6, hidden_size=64, num_layers=3)
# ... training loop ...

# Train ResNet model
from resnet import ResNetCBF, ResidualBlock
model = ResNetCBF(input_size=6, hidden_size=64, num_layers=3)
# ... training loop ...
```

### Training PPO Controller

```python
from ppo_controller import train_ppo
train_ppo(
    num_epochs=20,
    num_steps_per_epoch=2048
)
```

### Running Simulations

```python
from simulation import run

# Run with learned CBF (no RL)
run(model_path="path/to/cbf_model.pth", goal_idx=0, use_rl=False)

# Run with both learned CBF and RL controller
run(model_path="path/to/cbf_model.pth", goal_idx=0, use_rl=True)
```

## Key Algorithms

### Control Barrier Function (CBF)

For a dynamical system $\dot{x} = f(x) + g(x)u$, a function $h: \mathbb{R}^n \to \mathbb{R}$ is a CBF if:

$$\sup_{u \in \mathcal{U}} [L_f h(x) + L_g h(x) u] \geq -\alpha(h(x))$$

where $\alpha$ is an extended class $\mathcal{K}$ function. This ensures forward invariance of the safe set $\{x : h(x) \geq 0\}$.

### Analytical Barrier Function

For circular obstacles, the barrier function is:
$$h(x) = \min_i (||x - c_i|| - r_i)$$

where $c_i$ and $r_i$ are the center and radius of obstacle $i$.

### PPO Objective

$$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio.

## Dependencies

- PyTorch
- NumPy
- Pandas
- Matplotlib
- Gymnasium (Gym)
- tqdm
- CVXPY (for QP solver in trajectory optimization)

## Results

The project demonstrates:
- Successful learning of CBFs from analytical data
- Safe trajectory generation using learned CBFs
- RL controller capable of imitating nominal straight-line behavior
- Integration of both approaches for safe, goal-directed navigation

## Future Work

- Extend to higher-dimensional systems
- Learn CBFs directly from demonstrations without analytical supervision
- Combine CBF and RL in a unified framework
- Handle dynamic obstacles
- Multi-agent coordination with CBFs

## References

- Ames, A. D., et al. "Control barrier functions: Theory and applications." (2019)
- Schulman, J., et al. "Proximal policy optimization algorithms." (2017)
