"""
dueling_dqn_policy.py — Dueling architecture for SB3's DQN
============================================================
Stable-Baselines3's built-in DQNPolicy has no native `dueling` kwarg.
This module implements a dueling Q-network: shared feature extractor,
then two separate heads —
    V(s)      : scalar state-value stream
    A(s, a)   : per-action advantage stream
combined as:
    Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))

Use DuelingDQNPolicy in place of "MlpPolicy" wherever a dueling
architecture is wanted (Dueling DQN, Dueling Double DQN — Double DQN
itself is already always-on in SB3's DQN via the target network, so
"Double" and "Dueling Double" only differ from "DQN"/"Dueling DQN" in
hyperparameters and the policy class, not in extra flags).
"""

import torch
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.torch_layers import FlattenExtractor


class DuelingQNetwork(QNetwork):
    """Q-network with separate value/advantage streams."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        action_dim = int(self.action_space.n)
        feat_dim   = self.features_dim

        # rebuild the head: drop the single Q-head from QNetwork's __init__
        # and replace it with dueling value/advantage streams.
        net_arch = self.net_arch if self.net_arch is not None else [64, 64]

        def mlp_trunk(input_dim, layers):
            modules = []
            last = input_dim
            for h in layers:
                modules += [nn.Linear(last, h), nn.ReLU()]
                last = h
            return nn.Sequential(*modules), last

        # shared trunk operates on extracted features
        self.shared_trunk, trunk_out = mlp_trunk(feat_dim, net_arch)

        self.value_head     = nn.Linear(trunk_out, 1)
        self.advantage_head = nn.Linear(trunk_out, action_dim)

        # the original self.q_net (built by QNetwork.__init__) is unused
        # now; remove its parameters from the optimizer's view by deleting
        # the attribute so it isn't double-counted.
        if hasattr(self, "q_net"):
            del self.q_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        h = self.shared_trunk(features)
        v = self.value_head(h)                       # (batch, 1)
        a = self.advantage_head(h)                    # (batch, n_actions)
        q = v + (a - a.mean(dim=1, keepdim=True))      # dueling combine
        return q


class DuelingDQNPolicy(DQNPolicy):
    """DQNPolicy variant that swaps in DuelingQNetwork for both the
    online and target Q-networks."""

    def make_q_net(self) -> DuelingQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)
