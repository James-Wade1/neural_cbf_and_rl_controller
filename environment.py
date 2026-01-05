import numpy as np
from obstacles import Obstacle
from controllers import ControllerAbstract, NominalController
import torch

class Robot:
    def __init__(self, radius: float, position: np.ndarray, controller: ControllerAbstract = NominalController(gain=1.0)):
        self.radius: float = radius
        self.position: np.ndarray = position
        self.color: str = "blue"
        self.controller: ControllerAbstract = controller

    def get_nominal_ctrl(self, goal_pos: np.ndarray) -> np.ndarray:
        """Return the nominal dynamics of the robot."""
        return self.controller.get_control(self.position, goal_pos)

    def get_position(self) -> np.ndarray:
        return self.position

    def step(self, u: np.ndarray, dt: float) -> None:
        self.position += u*dt + np.random.normal(0, 0.01, size=self.position.shape) * dt

class Environment:
    def __init__(self, robot: Robot, obstacles: list[Obstacle], goal: np.ndarray = None):
        self.robot: Robot = robot
        self.obstacles: list[Obstacle] = obstacles
        self.goal: np.ndarray = goal

    def set_goal(self, goal: np.ndarray):
        self.goal = goal

    def get_positional_error(self):
        if self.goal is None:
            raise Exception("Goal is undefined in Environment")
        return np.linalg.norm(self.robot.position - self.goal)

    def step(self, u: np.ndarray, dt: float) -> None:
        self.robot.step(u, dt)

    def get_robot_nominal_ctrl(self, goal: np.ndarray) -> np.ndarray:
        return self.robot.get_nominal_ctrl(goal)

class Visualizer:
    def __init__(self, env: Environment):
        self.env: Environment = env

    def render(self, traj: np.ndarray, anim: bool = False, title: str = "", save_path: str = None) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.patches import Circle, Polygon
        from matplotlib.transforms import Affine2D

        def create_robot_triangle(position, direction=None, size=None):
            """Create a triangle representing the robot, pointing in direction of movement."""
            if size is None:
                size = self.env.robot.radius

            # Define triangle vertices (pointing upward by default)
            # Isosceles triangle with base at bottom
            base_vertices = np.array([
                [0, size * 1.5],      # Top vertex (front)
                [-size, -size],       # Bottom left
                [size, -size]         # Bottom right
            ])

            # Calculate rotation angle from direction
            if direction is not None and np.linalg.norm(direction) > 1e-6:
                angle = np.arctan2(direction[1], direction[0]) - np.pi/2  # -90 deg offset since default points up
            else:
                angle = 0

            # Rotate vertices
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_vertices = base_vertices @ rotation_matrix.T

            # Translate to position
            triangle_vertices = rotated_vertices + position

            return triangle_vertices

        fig, ax = plt.subplots(figsize=(10, 10))

        # Set up the plot limits and appearance
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 7)
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Robot Trajectory with {title}", fontsize=14, fontweight='bold')
        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Y Position", fontsize=12)

        # Draw obstacles with labels
        for idx, obs in enumerate(self.env.obstacles):
            obs_circle = Circle(obs.center, obs.radius, color=obs.color, alpha=0.6, label=f'Obstacle {idx+1}')
            ax.add_patch(obs_circle)
            # Add barrier function visualization (safety margin)
            safety_circle = Circle(obs.center, obs.radius * 1.5, color=obs.color, alpha=0.15, linestyle='--', fill=False)
            ax.add_patch(safety_circle)

        # Draw goal
        if self.env.goal is not None:
            goal_circle = Circle(self.env.goal, 0.2, color="gold", label='Goal', zorder=10)
            ax.add_patch(goal_circle)
            # Add a star marker at goal
            ax.plot(self.env.goal[0], self.env.goal[1], marker='*', markersize=20, color='gold', markeredgecolor='orange', markeredgewidth=2)

        if anim:
            # Create animated version with triangle robot
            # Calculate initial direction
            if len(traj) > 1:
                initial_direction = traj[1] - traj[0]
            else:
                initial_direction = np.array([0, 1])

            robot_vertices = create_robot_triangle(traj[0], initial_direction)
            robot_triangle = Polygon(robot_vertices, color=self.env.robot.color, label='Robot', zorder=5)
            ax.add_patch(robot_triangle)

            # Plot trajectory path (will be updated)
            trajectory_line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=2, label='Path')
            trajectory_points, = ax.plot([], [], 'bo', markersize=4, alpha=0.3)

            # Add start position marker
            ax.plot(traj[0, 0], traj[0, 1], marker='s', markersize=12, color='blue', markeredgecolor='darkblue', markeredgewidth=2, label='Start', zorder=10)

            # Text for step counter
            step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.legend(loc='upper right', fontsize=10)

            def init():
                trajectory_line.set_data([], [])
                trajectory_points.set_data([], [])
                return robot_triangle, trajectory_line, trajectory_points, step_text

            def update(frame):
                # Calculate direction for this frame
                if frame < len(traj) - 1:
                    direction = traj[frame + 1] - traj[frame]
                elif frame > 0:
                    direction = traj[frame] - traj[frame - 1]
                else:
                    direction = np.array([0, 1])

                # Update robot triangle position and orientation
                new_vertices = create_robot_triangle(traj[frame], direction)
                robot_triangle.set_xy(new_vertices)

                # Update trajectory
                trajectory_line.set_data(traj[:frame+1, 0], traj[:frame+1, 1])
                trajectory_points.set_data(traj[:frame+1, 0], traj[:frame+1, 1])

                # Update step counter
                distance_to_goal = np.linalg.norm(traj[frame] - self.env.goal) if self.env.goal is not None else 0
                step_text.set_text(f'Step: {frame}/{len(traj)-1}\nDistance to goal: {distance_to_goal:.3f}')

                return robot_triangle, trajectory_line, trajectory_points, step_text

            # Create animation with smooth playback
            anim_obj = FuncAnimation(fig, update, frames=len(traj), init_func=init,
                                     blit=True, interval=50, repeat=True)

            # Save animation as MP4 if save_path is provided
            if save_path:
                print(f"Saving animation to {save_path}...")
                try:
                    # Try using ffmpeg writer (preferred for MP4)
                    from matplotlib.animation import FFMpegWriter
                    writer = FFMpegWriter(fps=20, metadata=dict(artist='Robot Simulation'), bitrate=1800)
                    anim_obj.save(save_path, writer=writer)
                    print(f"Animation saved successfully to {save_path}")
                except Exception as e:
                    print(f"Error saving with FFMpegWriter: {e}")
                    print("Trying alternative writer...")
                    try:
                        # Fallback to pillow writer
                        anim_obj.save(save_path, writer='pillow', fps=20)
                        print(f"Animation saved successfully to {save_path}")
                    except Exception as e2:
                        print(f"Error saving animation: {e2}")
                        print("Make sure ffmpeg is installed or try a different format.")

            plt.show()
        else:
            # Static version showing full trajectory with triangle robot
            # Calculate final direction
            if len(traj) > 1:
                final_direction = traj[-1] - traj[-2]
            else:
                final_direction = np.array([0, 1])

            robot_vertices = create_robot_triangle(self.env.robot.position, final_direction)
            robot_triangle = Polygon(robot_vertices, color=self.env.robot.color, label='Robot (final)', zorder=5)
            ax.add_patch(robot_triangle)

            # Plot full trajectory
            ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=2, label='Path')
            ax.plot(traj[:, 0], traj[:, 1], 'bo', markersize=4, alpha=0.3)

            # Mark start and end
            ax.plot(traj[0, 0], traj[0, 1], marker='s', markersize=12, color='blue', markeredgecolor='darkblue', markeredgewidth=2, label='Start', zorder=10)
            ax.plot(traj[-1, 0], traj[-1, 1], marker='o', markersize=12, color='blue', markeredgecolor='darkblue', markeredgewidth=2, label='End', zorder=10)

            ax.legend(loc='upper right', fontsize=10)
            plt.show()

    def plot_min_h_values(self, grid_resolution: int = 50, xlim: tuple = (-1, 6), ylim: tuple = (-1, 7), title: str = "") -> None:
        """
        Plot the minimum h value across all obstacles at grid points in the environment.
        Creates a color map showing the range of barrier function values.

        Args:
            grid_resolution: Number of grid points in each dimension
            xlim: Tuple of (xmin, xmax) for plot limits
            ylim: Tuple of (ymin, ymax) for plot limits
            title: Title for the plot
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        # Create grid
        x = np.linspace(xlim[0], xlim[1], grid_resolution)
        y = np.linspace(ylim[0], ylim[1], grid_resolution)
        X, Y = np.meshgrid(x, y)

        # Compute minimum h value at each grid point
        with torch.no_grad():
            H = np.zeros((grid_resolution, grid_resolution))
            for i in range(grid_resolution):
                for j in range(grid_resolution):
                    point = np.array([X[i, j], Y[i, j]])
                    # Compute h value for each obstacle and take minimum
                    h_values = [obs.h(point) for obs in self.env.obstacles]
                    H[i, j] = min(h_values)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Determine the range
        h_min = np.min(H)
        h_max = np.max(H)

        # Create custom normalization: Red (h<0), Yellow (0<=h<=1), Green (h>1)
        from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, TwoSlopeNorm
        import matplotlib.colors as mcolors

        # Define color transitions at specific h values
        # We want: Red for h<0, Yellow for 0<=h<=1, Green for h>1
        boundaries = [h_min, 0, 1, h_max]
        colors = ['red', 'yellow', 'green']
        n_bins = 100  # Number of discrete colors
        cmap = LinearSegmentedColormap.from_list('custom_RdYlGn',
                                                  [(0, 'red'), (0.5, 'yellow'), (1, 'green')],
                                                  N=n_bins)

        # Create normalization that maps h values to colormap positions
        # Map: h_min->0, 0->0.5*(1/(1-h_min/h_max)), 1->something, h_max->1
        if h_max > 1:
            norm = mcolors.Normalize(vmin=h_min, vmax=h_max)
            # Manually adjust the mapping
            class CustomNorm(mcolors.Normalize):
                def __call__(self, value, clip=None):
                    # Map h_min to 0, 0 to 1/3, 1 to 2/3, h_max to 1
                    result = np.ma.masked_array(np.zeros_like(value))
                    # For h < 0: map linearly from h_min->0 to 0->1/3
                    mask_negative = value < 0
                    if h_min < 0:
                        result[mask_negative] = (value[mask_negative] - h_min) / (0 - h_min) / 3
                    # For 0 <= h <= 1: map linearly from 0->1/3 to 1->2/3
                    mask_transition = (value >= 0) & (value <= 1)
                    result[mask_transition] = 1/3 + (value[mask_transition] - 0) / (1 - 0) / 3
                    # For h > 1: map linearly from 1->2/3 to h_max->1
                    mask_positive = value > 1
                    if h_max > 1:
                        result[mask_positive] = 2/3 + (value[mask_positive] - 1) / (h_max - 1) / 3
                    return result
            norm = CustomNorm(vmin=h_min, vmax=h_max)
        else:
            # If h_max <= 1, use simpler normalization
            norm = TwoSlopeNorm(vmin=h_min, vcenter=0.5, vmax=1)

        # Plot color map with custom normalization
        contour = ax.contourf(X, Y, H, levels=100, cmap=cmap, alpha=0.8, norm=norm)

        # Add contour lines at h=0 and h=1
        contour_lines_0 = ax.contour(X, Y, H, levels=[0], colors='black', linewidths=2, linestyles='--')
        ax.clabel(contour_lines_0, inline=True, fontsize=10, fmt='h=%.1f')
        contour_lines_1 = ax.contour(X, Y, H, levels=[1], colors='darkgreen', linewidths=1.5, linestyles=':')
        ax.clabel(contour_lines_1, inline=True, fontsize=9, fmt='h=%.1f')

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Minimum h value (barrier function)', fontsize=12, fontweight='bold')

        # Draw obstacles
        for idx, obs in enumerate(self.env.obstacles):
            obs_circle = Circle(obs.center, obs.radius, color='black', alpha=0.3,
                              edgecolor='black', linewidth=2, label=f'Obstacle {idx+1}' if idx == 0 else '')
            ax.add_patch(obs_circle)
            # Add text label
            ax.text(obs.center[0], obs.center[1], f'{idx+1}',
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Set up the plot
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_title(f"Barrier Function Heat Map{' - ' + title if title else ''}",
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Y Position", fontsize=12)

        # Add text box explaining the color map
        textstr = 'Green: Safe (h > 1)\nYellow: Transition (0 ≤ h ≤ 1)\nRed: Unsafe (h < 0)'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.show()

    def plot_velocity_vector_field(self, goal: np.ndarray, grid_resolution: int = 20,
                                   xlim: tuple[float, float] = (-1, 6),
                                   ylim: tuple[float, float] = (-1, 7),
                                   title: str = "") -> None:
        """
        Plot a vector field showing velocity arrows from robot positions to the goal.
        Velocity is calculated using the robot's controller.

        Args:
            goal: Goal position as numpy array [x, y]
            grid_resolution: Number of grid points in each dimension
            xlim: X-axis limits (min, max)
            ylim: Y-axis limits (min, max)
            title: Title suffix for the plot
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        # Create grid
        x = np.linspace(xlim[0], xlim[1], grid_resolution)
        y = np.linspace(ylim[0], ylim[1], grid_resolution)
        X, Y = np.meshgrid(x, y)

        # Initialize velocity components
        U = np.zeros((grid_resolution, grid_resolution))
        V = np.zeros((grid_resolution, grid_resolution))

        # Store original robot position
        original_position = self.env.robot.position.copy()

        # Compute velocity at each grid point
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                point = np.array([X[i, j], Y[i, j]])
                # Temporarily set robot position to grid point
                self.env.robot.position = point
                # Get velocity from robot's controller
                velocity = self.env.robot.get_nominal_ctrl(goal)
                U[i, j] = velocity[0]
                V[i, j] = velocity[1]

        # Restore original robot position
        self.env.robot.position = original_position

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot vector field
        # Normalize arrow lengths for better visualization
        magnitude = np.sqrt(U**2 + V**2)
        max_magnitude = np.max(magnitude)
        if max_magnitude > 0:
            scale = 1.0 / max_magnitude * 3  # Scale factor for arrow lengths
        else:
            scale = 1.0

        # Create quiver plot
        quiver = ax.quiver(X, Y, U, V, magnitude,
                          cmap='viridis',
                          scale=1/scale*15,
                          scale_units='xy',
                          angles='xy',
                          width=0.003,
                          alpha=0.8)

        # Add colorbar for magnitude
        cbar = plt.colorbar(quiver, ax=ax)
        cbar.set_label('Velocity Magnitude', fontsize=12, fontweight='bold')

        # Draw obstacles
        for idx, obs in enumerate(self.env.obstacles):
            obs_circle = Circle(obs.center, obs.radius, color=obs.color, alpha=0.5,
                              edgecolor='black', linewidth=2, label=f'Obstacle {idx+1}')
            ax.add_patch(obs_circle)

        # Draw goal position
        ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal', markeredgecolor='black', markeredgewidth=1.5)

        # Draw starting position if defined
        if self.env.robot is not None:
            ax.plot(original_position[0], original_position[1], 'bo', markersize=12,
                   label='Robot Start', markeredgecolor='black', markeredgewidth=1.5)

        # Set up the plot
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_title(f"Velocity Vector Field to Goal{' - ' + title if title else ''}",
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Y Position", fontsize=12)
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.show()
