import os
import time
import argparse

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_factory import make_env, make_vec_env
from metrics_callback import MetricsCallback

class SparseA2C(A2C):

    def __init__(self, *args, update_every_k=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_every_k = int(update_every_k)
        self._rollout_idx = 0

    def train(self) -> None:
        self._rollout_idx += 1
        if self.update_every_k > 1 and (self._rollout_idx % self.update_every_k != 0):
            self.logger.record("sparse/skipped_update", 1)
            return
        self.logger.record("sparse/skipped_update", 0)
        super().train()


def main():
    parser = argparse.ArgumentParser(description="Run Sparse A2C experiments")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for k in [1, 4, 8]:
        print(f"\n--- Running Sparse A2C with K={k} ---")

        best_model_dir = f"./models/best_sparse_a2c_k{k}/"
        eval_log_dir = f"./logs/sparse_a2c_k{k}_eval/"
        checkpoint_dir = f"./checkpoints/sparse_a2c_k{k}/"
        tb_log_dir = f"./logs/sparse_a2c_k{k}/"

        for d in [best_model_dir, eval_log_dir, checkpoint_dir]:
            os.makedirs(d, exist_ok=True)

        train_env = make_vec_env(
            n_envs=8,
            seed=args.seed,
            use_subprocess=False,
            norm_reward=True,
        )

        eval_env = VecNormalize(
            DummyVecEnv([make_env(rank=100, seed=args.seed)]),
            norm_obs=True,
            norm_reward=False,
            clip_obs=5.0,
            gamma=0.99,
        )
        eval_env.training = False

        model = SparseA2C(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=512,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            tensorboard_log=tb_log_dir,
            device=args.device,
            seed=args.seed,
            verbose=1,
            update_every_k=k,
        )

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            log_path=eval_log_dir,
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
        )
        ckpt_cb = CheckpointCallback(save_freq=100_000, save_path=checkpoint_dir)
        metrics_cb = MetricsCallback()

        t0 = time.perf_counter()

        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=[eval_cb, ckpt_cb, metrics_cb],
                reset_num_timesteps=True,
            )
        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] K={k} — saving partial model")

        wall = time.perf_counter() - t0

        model.save(f"./models/sparse_a2c_k{k}")
        train_env.save(f"./models/vecnormalize_sparse_a2c_k{k}.pkl")

        print(f"K={k}  wall={wall:.1f}s ({wall/60:.1f} min)")

        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()