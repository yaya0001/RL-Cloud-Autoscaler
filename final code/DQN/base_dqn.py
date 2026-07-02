"""
BaseDQN Middle Parent Class
===========================

Sits between SB3's DQN and the four concrete variants:

    SB3 DQN
        └── BaseDQN                   (this file)
                ├── VanillaDQN
                ├── DoubleDQN
                ├── DuelingDQN
                └── DoubleDuelingDQN

Responsibility of this class
-----------------------------
1. Own the update_frequency attribute (the sparse-update ablation variable).
2. Override SB3's train() to enforce the update frequency gate.
3. Declare _compute_td_target() as the hook variants override for
   target computation changes (Double DQN, Double+Dueling DQN).
4. Declare _build_network() as the hook variants override for
   architecture changes (Dueling DQN, Double+Dueling DQN).

"""

import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.type_aliases import GymEnv
from typing import Any, Dict, List, Optional, Tuple, Type, Union


class BaseDQN(DQN):
    """Middle parent class for all four DQN variants.

    Adds update_frequency gating on top of SB3's DQN and declares
    the two hook methods that concrete variants selectively override.

    Parameters
    ----------
    policy : str or DQNPolicy subclass
        Passed through to SB3's DQN unchanged.
    env : GymEnv
        Training environment, already wrapped in VecNormalize.
    update_frequency : int
        How many environment steps to collect between gradient updates.
        1 = update every step.
        2 = update every 2 steps.
        4 = update every 4 steps.
        8 = update every 8 steps.
    **kwargs
        All remaining keyword arguments forwarded to SB3's DQN.__init__
        unchanged (learning_rate, buffer_size, batch_size, etc.).
    """

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: GymEnv,
        update_frequency: int = 4,
        **kwargs,
    ):
        # first set the update_frequency to implemet the sparse ablation
        # then send everything else to SB3's DQN constructor
        self.update_frequency = update_frequency
        super().__init__(policy=policy, env=env, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Override SB3's train() to enforce update_frequency gating.

        Example with update_frequency = 4 and total_timesteps=1_000_000:
            step 1 → gate blocks  (1 % 4 != 0)
            step 2 → gate blocks  (2 % 4 != 0)
            step 3 → gate blocks  (3 % 4 != 0)
            step 4 → gate passes  → _compute_td_target() + backprop
            step 5 → gate blocks
            ...
        """
        if self.num_timesteps % self.update_frequency != 0:
            return   # sparse-update: skip this gradient step
        super().train(gradient_steps, batch_size) # if else gate passes

    # Hook #1: Compute the TD target for standard (Vanilla)
    def _compute_td_target(
        self,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the TD target for standard (Vanilla)
                y = r + γ · max_a Q_target(s', a).
        Parameters
        ----------
        next_observations : torch.Tensor, shape (batch, obs_dim)
            Next states s' sampled from the replay buffer.
        rewards : torch.Tensor, shape (batch,)
            Immediate rewards r from the replay buffer.
        dones : torch.Tensor, shape (batch,)
            Episode termination flags (1.0 if terminal, else 0.0).

        """
        with torch.no_grad():
            # target network evaluates all actions at s'
            next_q_values = self.q_net_target(next_observations)
            # take the max
            next_q_values, _ = next_q_values.max(dim=1)
            # mask terminal states: if done, no future reward
            next_q_values = next_q_values * (1.0 - dones)

        return rewards + self.gamma * next_q_values

    # Hook #2: Build Network Architecture
    def _build_network(self) -> nn.Module:
        """Return the Q-network architecture as an nn.Module.

        The default implementation returns a standard MLP identical
        to SB3's built-in QNetwork, which produces Q-values directly:

            obs (5,) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(3)

        DuelingDQN and DoubleDuelingDQN override this to return a
        network whose forward() splits into value + advantage heads:

            obs (5,) → Linear(256) → ReLU → Linear(256) → ReLU ─┬→ Linear(256→1)  = V(s)
                                                                  └→ Linear(256→3)  = A(s,a)
        """
        # Returns None in the base because SB3 owns network construction.
        return None

    # log function
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"update_frequency={self.update_frequency}, "
            f"lr={self.learning_rate}, "
            f"buffer={self.buffer_size}, "
            f"batch={self.batch_size})"
        )