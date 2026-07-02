import optuna
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from env_factory import make_vec_env




def objective(trial):

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-3,
        log=True
    )

    clip_range = trial.suggest_float(
        "clip_range",
        0.1,
        0.4
    )

    ent_coef = trial.suggest_float(
        "ent_coef",
        1e-5,
        1e-1,
        log=True
    )

    gamma = trial.suggest_float(
        "gamma",
        0.95,
        0.9999
    )

    gae_lambda = trial.suggest_float(
        "gae_lambda",
        0.90,
        0.99
    )

    vf_coef = trial.suggest_float(
        "vf_coef",
        0.1,
        1.0
    )

    max_grad_norm = trial.suggest_float(
        "max_grad_norm",
        0.3,
        1.0
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [64, 128, 256, 512]
    )

    n_steps = trial.suggest_categorical(
        "n_steps",
        [512, 1024, 2048, 4096]
    )

    n_epochs = trial.suggest_categorical(
        "n_epochs",
        [5, 10, 15]
    )

    

    env = make_vec_env(
        n_envs=8,
        seed=42,
        use_subprocess=True,
        norm_reward=True
    )

    

    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=learning_rate,

        n_steps=n_steps,

        batch_size=batch_size,

        n_epochs=n_epochs,

        gamma=gamma,

        gae_lambda=gae_lambda,

        clip_range=clip_range,

        ent_coef=ent_coef,

        vf_coef=vf_coef,

        max_grad_norm=max_grad_norm,

        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256]
            )
        ),

        device="auto",

        verbose=0
    )


    model.learn(
        total_timesteps=100_000
    )

    
    mean_reward, _ = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True
    )

    env.close()

    return mean_reward



if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        study_name="ppo_cloud_autoscaling"
    )

    study.optimize(
        objective,
        n_trials=50,
        show_progress_bar=True
    )

    print("BEST TRIAL")
    print("==============================")

    print(f"Best Reward: {study.best_value}")

    print("\nBest Parameters:\n")

    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    print("\n==============================")