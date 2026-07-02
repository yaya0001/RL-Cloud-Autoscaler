from base_dqn import BaseDQN


class VanillaDQN(BaseDQN):
    """Vanilla DQN implemets the standard target, standard MLP and there are no modifications.

    Overrides nothing from BaseDQN. All behavior is inherited.

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
    LABEL = "Vanilla DQN"
    SLUG  = "vanilla_dqn"

    # Output path templates formatted with update_frequency at runtime.
    # Example: PATHS["log_dir"].format(freq=4) → "./logs/vanilla_dqn_freq4/"
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
            policy="MlpPolicy",    # standard architecture — no custom policy needed
            env=env,
            update_frequency=update_frequency,
            **kwargs,
        )

    # Return fully formatted output paths for a given update_frequency.
    @classmethod
    def get_paths(cls, update_frequency: int) -> dict:
        return {
            key: template.format(slug=cls.SLUG, freq=update_frequency)
            for key, template in cls.PATHS.items()
        }