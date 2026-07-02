"""
Why the decomposition helps
----------------------------
The problem :: That final Linear layer has three separate output neurons, one per action. Each neuron has its own weights. When you take a gradient step, only the neuron corresponding to the action you actually took gets updated.
In CloudScaling-v1, most timesteps the right action is "Hold".
The value of the STATE (how loaded the system is) dominates the choice instead of the taken actions.


Dueling DQN updates V(s) on EVERY step regardless of which action was taken.
V(s) is shared across all three action Q-values.
This makes the network learn state values much faster especially during the quiet sinusoidal troughs where "Hold" dominates and standard DQN barely learns from those steps.

What this class overrides
--------------------------
- _build_network() → documented hook in BaseDQN, satisfied here by passing DuelingDQNPolicy to  __init__ rather than "MlpPolicy"
- DuelingQNetwork → replaces SB3's QNetwork forward()
- DuelingDQNPolicy → tells SB3 to use DuelingQNetwork for both online and target networks

"""

import torch
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

from base_dqn import BaseDQN


class DuelingQNetwork(QNetwork):
    """Dueling Q-network: Value stream + Advantage stream.

    Subclasses SB3's QNetwork. The parent __init__ builds the full
    standard MLP in self.q_net as an nn.Sequential:

        [Linear(5→256), ReLU, Linear(256→256), ReLU, Linear(256→3)]

    We keep everything except the last Linear and replace it with
    two separate heads.

    Parameters
    ----------
    All parameters are forwarded to QNetwork.__init__ unchanged.
    DuelingDQNPolicy handles passing the right arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = list(self.q_net.children())
        trunk_layers = layers[:-1]
        last_linear  = layers[-1]
        in_features = last_linear.in_features

        # Rebuild self.q_net as trunk only
        self.q_net = nn.Sequential(*trunk_layers)
        # value_head: How good is this state?
        self.value_head = nn.Linear(in_features, 1)
        # advantage_head: How much better is each action compared to the average action in this state?
        self.advantage_head = nn.Linear(in_features, self.action_space.n)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Compute Q(s,a) via the Dueling decomposition.

        # Step 1 SB3 preprocessing (normalisation, flattening)
        features = self.extract_features(obs, self.features_extractor)
        # Step 2 shared trunk
        trunk_out = self.q_net(features)
        # Step 3 two heads from the same trunk output
        V = self.value_head(trunk_out)
        A = self.advantage_head(trunk_out)
        # Step 4 Dueling combination with mean subtraction
        Q = V + (A - A.mean(dim=1, keepdim=True))

        return Q


#  Policy wrapper that tells SB3 to use DuelingQNetwork into SB3's policy system
class DuelingDQNPolicy(DQNPolicy):
    q_net_class = DuelingQNetwork


class DuelingDQN(BaseDQN):
    """Dueling DQN — Dueling architecture, standard TD target.

    Passes DuelingDQNPolicy to BaseDQN.__init__ instead of "MlpPolicy".
    This satisfies the _build_network() hook documented in BaseDQN:
    the architecture change is injected through the policy class,
    which is how SB3 exposes network customisation.

    Everything else — TD target, update_frequency gate, replay buffer,
    optimizer, exploration, target sync are all inherited from BaseDQN
    and SB3's DQN unchanged.

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

    LABEL = "Dueling DQN"
    SLUG  = "dueling_dqn"

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
            policy=DuelingDQNPolicy,   # the new policy is injected here
            env=env,
            update_frequency=update_frequency,
            **kwargs,
        )


    @classmethod
    def get_paths(cls, update_frequency: int) -> dict:
        return {
            key: template.format(slug=cls.SLUG, freq=update_frequency)
            for key, template in cls.PATHS.items()
        }