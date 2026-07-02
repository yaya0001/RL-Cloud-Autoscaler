"""

Why combining both helps beyond either alone
---------------------------------------------
DuelingDQN alone: learns state values faster (V(s) updated every
step), but the TD target is still overestimated because argmax
is applied to the target network's own Q-values.

DoubleDQN alone: fixes the overestimation bias in the TD target,
but still wastes learning signal on steps where the taken action
is "Hold" and the other two actions' Q-values sit frozen.

What this class overrides
--------------------------
- DoubleDuelingDQNPolicy → identical to DuelingDQNPolicy in effect, separate class for naming clarity
- train()
"""

from stable_baselines3 import DQN as SB3DQN
from stable_baselines3.dqn.policies import DQNPolicy

from base_dqn import BaseDQN
from dueling_dqn import DuelingQNetwork   # reuse — no duplication

import numpy as np
import torch as th
import torch.nn.functional as F


# Policy that builds both online and target networks as DuelingQNetwork.
class DoubleDuelingDQNPolicy(DQNPolicy):
    q_net_class = DuelingQNetwork

# Dueling architecture + Double-Q target.
class DoubleDuelingDQN(BaseDQN):
    """
    Parameters
    ----------
    env : GymEnv
        Training environment wrapped in VecNormalize.
    update_frequency : int
        Steps between gradient updates. 1/2/4/8 for the ablation.
        Default 4 matches the existing train_dqn.py configuration.
    **kwargs
        Forwarded to BaseDQN → SB3's DQN.__init__ unchanged.
        Pass SHARED_HYPERPARAMS here as **SHARED_HYPERPARAMS.
    """

    LABEL = "Double + Dueling DQN"
    SLUG  = "double_dueling_dqn"

    PATHS = {
        "log_dir": "./logs/{slug}_freq{freq}/",
        "eval_log": "./logs/{slug}_freq{freq}_eval/",
        "best_model": "./models/best_{slug}_freq{freq}/",
        "checkpoint": "./checkpoints/{slug}_freq{freq}/",
        "final_model": "./models/final_{slug}_freq{freq}",
        "vecnorm": "./models/vecnormalize_{slug}_freq{freq}.pkl",
    }

    def __init__(self, env, update_frequency: int = 4, **kwargs):
        super().__init__(
            policy=DoubleDuelingDQNPolicy,  # Dueling architecture injected
            env=env,
            update_frequency=update_frequency,
            **kwargs,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if self.num_timesteps % self.update_frequency != 0:
            return

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with th.no_grad():
                # Step 1: online Dueling network scores all actions at s'
                online_next_q = self.q_net(replay_data.next_observations)
                # Step 2: online network picks the best action (less biased)
                best_actions = online_next_q.argmax(dim=1, keepdim=True)
                # Step 3: target Dueling network scores all actions at s'
                target_next_q = self.q_net_target(replay_data.next_observations)
                # Step 4: target network evaluates only the online-selected action
                next_q_values = target_next_q.gather(dim=1, index=best_actions)
                # Step 5: mask terminals and compute TD target
                target_q_values = (
                        replay_data.rewards
                        + (1 - replay_data.dones) * discounts * next_q_values
                )

            # current Q for the actions actually taken (Dueling forward)
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            # Huber loss (same as SB3)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    # Return fully formatted output paths for a given update_frequency.
    @classmethod
    def get_paths(cls, update_frequency: int) -> dict:
        return {
            key: template.format(slug=cls.SLUG, freq=update_frequency)
            for key, template in cls.PATHS.items()
        }