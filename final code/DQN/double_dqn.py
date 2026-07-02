"""
Double DQN target:
    y = r + γ · Q_target(s', argmax_a Q_online(s', a))
                          └─── online net selects ────┘
                └─────── target net evaluates ─────────┘
    Fix: the online network picks the action (less biased
    because it is being actively trained and regularized),
    the target network scores it (stable, updated slowly).
    The two networks disagree slightly — this disagreement
    cancels the upward bias.

Why this kind of varient is expected to matter
----------------------------------------------
The problem :: At the start of training, the Q-network is a randomly initialized neural network. It has never seen a real episode.
So its Q-values for any state are essentially noise where some actions get slightly high estimates by pure chance, some get slightly low ones.

The reward function has extreme outliers.
At a given timestep where requests are dropped gives reward ≈ -1500 while a normal traffic timestep gives ≈ -3.
Vanilla DQN sees a few lucky "hold" actions during light traffic that happen to avoid drops and overestimates how good "hold" actually is.
This makes it slow to scale out when a spike starts.

On the other hand Double DQN is more learns that "hold" is only good when traffic is low not good at all situations.
The difference is which network does which job.
Vanilla DQN: the target network both selects the best action and evaluates its value. Same network, same weights, same biases.
Double DQN: the online network selects which action looks best while another network (target network) evaluates how good that action actually is.

What this class overrides
--------------------------
- _compute_td_target() → Implements the Double-Q formula
- train() → A mechanism that makes _compute_td_target() take effect inside SB3's training loop

"""

import torch
from base_dqn import BaseDQN
import numpy as np
import torch as th
import torch.nn.functional as F

class DoubleDQN(BaseDQN):
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

    LABEL = "Double DQN"
    SLUG  = "double_dqn"

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
            policy="MlpPolicy",
            env=env,
            update_frequency=update_frequency,
            **kwargs,
        )

    #  Override the training to make _compute_td_target work
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
                # Step 1: online network scores all actions at s'
                online_next_q = self.q_net(replay_data.next_observations)
                # Step 2: online network picks the best action
                best_actions = online_next_q.argmax(dim=1, keepdim=True)
                # Step 3: target network scores all actions at s'
                target_next_q = self.q_net_target(replay_data.next_observations)
                # Step 4: target network evaluates only the online-selected action
                next_q_values = target_next_q.gather(dim=1, index=best_actions)
                # Step 5: mask terminals and compute TD target
                target_q_values = (
                        replay_data.rewards
                        + (1 - replay_data.dones) * discounts * next_q_values
                )

            # current Q for the actions actually taken
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


    @classmethod
    def get_paths(cls, update_frequency: int) -> dict:
        return {
            key: template.format(slug=cls.SLUG, freq=update_frequency)
            for key, template in cls.PATHS.items()
        }