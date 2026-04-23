from .base_env import BaseEnv
from .base_task_config import BaseTaskConfig
import numpy as np
import gym
from typing import Tuple

class TaskConfigCartpole(BaseTaskConfig):
    """Configuration for the cartpole task"""
    def __init__(self, task_id: int, task_vec):
        super().__init__(task_id, task_vec)
        self.task_id = task_id
        self.task_vec = task_vec

class Cartpole(BaseEnv):
    """Cartpole environment"""
    def __init__(self, task, max_steps: int = 1000):
        super().__init__(task=task)
        self.task = task
        self.env = gym.make("CartPole-v1").unwrapped
        # Network dimensions
        self.input_size = int(self.env.observation_space.shape[0])  # 4
        self.output_size = int(self.env.action_space.n)             # 2
        self.solution_dim = task.solution_dim
        self.hidden_size = (self.solution_dim - self.output_size) // (self.input_size + self.output_size + 1)
        self.max_steps = int(max_steps)

    def simulate(
        self,
        weights_input_hidden: np.ndarray,
        weights_hidden_output: np.ndarray,
        rollout_seed: int,
        mass_pole: float,
        pole_length: float,
        visualize: bool = False,
    ) -> float:
        """Run one rollout using the provided weights."""
        # Configure environment physics
        self.env.masspole = mass_pole
        self.env.length = pole_length
        self.env.total_mass = self.env.masspole + self.env.masscart
        self.env.polemass_length = self.env.masspole * self.env.length

        observation, _info = self.env.reset(seed=rollout_seed)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < self.max_steps:
            # Input with bias
            input_with_bias = np.ones((self.input_size + 1, 1), dtype=np.float64)
            input_with_bias[1:, 0] = observation

            # Hidden layer
            hidden_linear = weights_input_hidden @ input_with_bias
            hidden_activation = np.tanh(hidden_linear)

            # Hidden with bias
            hidden_with_bias = np.ones((self.hidden_size + 1, 1), dtype=np.float64)
            hidden_with_bias[1:, 0] = hidden_activation[:, 0]

            # Output layer (logits)
            output_logits = weights_hidden_output @ hidden_with_bias

            # Greedy action: 0 if first logit larger, else 1
            action = int(output_logits[1, 0] >= output_logits[0, 0])

            observation, reward, terminated, truncated, _info = self.env.step(action)
            total_reward += float(reward)
            steps += 1

            if visualize:
                self.env.render()

        return float(min(self.max_steps, total_reward))

    def evaluate_solution(
        self,
        flat_weights: np.ndarray,
        visualize: bool = False,
        n_rollouts: int = 10,
    ) -> float:
        """Evaluate a flat weight vector across multiple seeded rollouts."""
        mass_pole = self.task.task_vec[0]
        pole_length = self.task.task_vec[1]
        # Unflatten
        w1_size = self.hidden_size * (self.input_size + 1)
        w2_size = self.output_size * (self.hidden_size + 1)
        expected_size = w1_size + w2_size
        if flat_weights.size != expected_size:
            raise ValueError(
                f"flat_weights has size {flat_weights.size}, expected {expected_size} "
                f"for hidden_size={self.hidden_size}, input_size={self.input_size}, output_size={self.output_size}"
            )

        weights_input_hidden = flat_weights[:w1_size].reshape(self.hidden_size, self.input_size + 1)
        weights_hidden_output = flat_weights[w1_size:].reshape(self.output_size, self.hidden_size + 1)

        rewards = []
        for seed_idx in range(n_rollouts):
            reward = self.simulate(
                weights_input_hidden=weights_input_hidden,
                weights_hidden_output=weights_hidden_output,
                rollout_seed=seed_idx,
                mass_pole=mass_pole,
                pole_length=pole_length,
                visualize=visualize,
            )
            rewards.append(reward)

        return float(np.mean(rewards))

