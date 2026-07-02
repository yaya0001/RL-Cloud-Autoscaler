"""Train PPO on CloudScaling-v1 with EvalCallback, checkpoints, and custom metrics."""

import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cloud_env import CloudScalingEnv  # noqa: F401
from env_factory import make_env, make_vec_env
from metrics_callback import MetricsCallback


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on CloudScaling-v1")
    p.add_argument("--timesteps", type=int, default=2_000_000,
                   help="Total training timesteps (use 20000 for smoke test)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device for training (benchmark cpu vs cuda yourself)")
    return p.parse_args()


def main():
    args = parse_args()

    # create output dirs
    for d in ["./models/best_ppo", "./checkpoints/ppo",
              "./logs/ppo", "./logs/ppo_eval"]:
        os.makedirs(d, exist_ok=True)

    # training env: 8 parallel envs with SubprocVecEnv + VecNormalize
    train_env = make_vec_env(n_envs=8, seed=0, use_subprocess=True,
                             norm_reward=True)

    # eval env: single env, no reward normalization, frozen stats
    eval_env = VecNormalize(
        DummyVecEnv([make_env(rank=100, seed=0)]),
        norm_obs=True, norm_reward=False, clip_obs=5.0, gamma=0.99)
    eval_env.training = False

    # PPO model
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1.4377739650640294e-05,
        n_steps=4096,
        batch_size=64,
        n_epochs=5,
        gamma=0.9660969058740851,
        gae_lambda=0.9175042883882351,
        clip_range=0.1291937132179491,
        ent_coef=0.012344366981406195,
        vf_coef=0.9315525456344808,
        max_grad_norm=0.48170949108431693,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log="./logs/ppo/",
        device=args.device,
        verbose=1,
    )

    # callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_ppo/",
        log_path="./logs/ppo_eval/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(save_freq=100_000,
                                 save_path="./checkpoints/ppo/")
    metrics_cb = MetricsCallback()

    print("=" * 60)
    print(f"  [START] PPO Training")
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

    # save model + VecNormalize stats (always, even after Ctrl+C)
    model.save("./models/final_ppo")
    train_env.save("./models/vecnormalize_ppo.pkl")

    wall_time = time.perf_counter() - t0

    print()
    print(f"  [DONE] PPO Training")
    print(f"  Wall-clock time: {wall_time:.1f}s ({wall_time/60:.1f} min)")
    print(f"  Model saved to: ./models/final_ppo.zip")
    print(f"  VecNormalize saved to: ./models/vecnormalize_ppo.pkl")
    print(f"  Best model (EvalCallback): ./models/best_ppo/best_model.zip")
    print(f"  Eval logs: ./logs/ppo_eval/evaluations.npz")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
