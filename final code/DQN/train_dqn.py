"""Train DQN on CloudScaling-v1 with EvalCallback, checkpoints, and custom metrics."""

import argparse
import os
import time

from vanilla_dqn import VanillaDQN
from double_dqn import DoubleDQN
from dueling_dqn import DuelingDQN
from double_dueling_dqn import DoubleDuelingDQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cloud_env import CloudScalingEnv  # noqa: F401
from env_factory import make_env, make_vec_env
from metrics_callback import MetricsCallback

VARIANT_MAP = {
    "vanilla":        VanillaDQN,
    "double":         DoubleDQN,
    "dueling":        DuelingDQN,
    "double_dueling": DoubleDuelingDQN,
}

# Per-variant tuned hyperparameters, pulled from each variant's best Optuna trial (dqn_optimization_results.json, dueling_dqn_optimization_results.json).
VARIANT_HYPERPARAMS = {
    "vanilla": dict(
        learning_rate=8.42665034538258e-05,
        buffer_size=500_000,
        learning_starts=5_000,
        batch_size=128,
        tau=1.0,
        gamma=0.9907466921420781,
        train_freq=8,
        gradient_steps=16,
        target_update_interval=1000,
        exploration_fraction=0.21573775929604358,
        exploration_initial_eps=0.9485296583267736,
        exploration_final_eps=0.025018757601193223,
        policy_kwargs=dict(net_arch=[128, 256, 128]),
    ),
    "double": dict(
        learning_rate=0.00010927879883233762,
        buffer_size=1_000_000,
        learning_starts=1_000,
        batch_size=256,
        tau=1.0,
        gamma=0.9654786631566694,
        train_freq=8,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.12507445265541792,
        exploration_initial_eps=0.9292317317572214,
        exploration_final_eps=0.08358632912993018,
        policy_kwargs=dict(net_arch=[512, 512]),
    ),
    # TODO: replace with tuned values once the dueling-family
    "dueling": dict(
        learning_rate=0.0007746859494512188,
        buffer_size=1_000_000,
        learning_starts=5_000,
        batch_size=64,
        tau=1.0,
        gamma=0.9502843111169749,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.4595670006588379,
        exploration_initial_eps=0.8167358365079543,
        exploration_final_eps=0.0995978490043962,
        policy_kwargs=dict(net_arch=[256, 512, 256]),
    ),
    "double_dueling": dict(
        learning_rate=0.00048287152161792117,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        tau=1.0,
        gamma=0.9578998430754462,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.191460191484347,
        exploration_initial_eps=0.7542853455823514,
        exploration_final_eps=0.09168098265334837,
        policy_kwargs=dict(net_arch=[512, 512]),
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Train DQN on CloudScaling-v1")
    p.add_argument("--timesteps", type=int, default=2_000_000,
                   help="Total training timesteps (use 20000 for smoke test)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device for training")
    p.add_argument("--variant", default="vanilla", choices=list(VARIANT_MAP.keys()),
                   help="Algorithm varients for training (Specifically used for DQN)")
    p.add_argument("--update_frequency", type=int, default=4, choices=[1, 2, 4, 8],
                   help="Steps between gradient updates (sparse ablation)")
    return p.parse_args()


def main():
    args = parse_args()

    AgentClass = VARIANT_MAP[args.variant]
    hyperparams = VARIANT_HYPERPARAMS[args.variant]
    paths = AgentClass.get_paths(args.update_frequency)

    for key in ["best_model", "checkpoint", "log_dir", "eval_log"]:
        os.makedirs(paths[key], exist_ok=True)

    # single env with DummyVecEnv -- never SubprocVecEnv for DQN
    train_env = make_vec_env(n_envs=1, seed=0, use_subprocess=False,
                             norm_reward=True)

    # eval env: no reward normalization, frozen stats
    eval_env = VecNormalize(
        DummyVecEnv([make_env(rank=100, seed=0)]),
        norm_obs=True, norm_reward=False, clip_obs=5.0, gamma=0.99)
    eval_env.training = False

    model = AgentClass(
        env=train_env,
        update_frequency=args.update_frequency,
        tensorboard_log=paths["log_dir"],
        device=args.device,
        verbose=1,
        **hyperparams,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=paths["best_model"],
        log_path=paths["eval_log"],
        eval_freq=80_000,
        n_eval_episodes=5,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(save_freq=100_000,
                                 save_path=paths["checkpoint"])
    metrics_cb = MetricsCallback()

    print("=" * 60)
    print(f"  [START] {AgentClass.LABEL} Training")
    print(f"  Variant: {args.variant} | Freq: {args.update_frequency}")
    print(f"  Device: {args.device} | Timesteps: {args.timesteps:,}")
    print("=" * 60)

    t0 = time.perf_counter()

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_cb, ckpt_cb, metrics_cb],
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving model before exit ...")

    model.save(paths["final_model"])
    train_env.save(paths["vecnorm"])

    wall_time = time.perf_counter() - t0

    print()
    print("=" * 60)
    print(f"  [DONE] {AgentClass.LABEL} Training")
    print(f"  Wall-clock time: {wall_time:.1f}s ({wall_time / 60:.1f} min)")
    print(f"  Model saved to:  {paths['final_model']}.zip")
    print(f"  VecNormalize:    {paths['vecnorm']}")
    print(f"  Best model:      {paths['best_model']}best_model.zip")
    print(f"  Eval logs:       {paths['eval_log']}evaluations.npz")
    print("=" * 60)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
