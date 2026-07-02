import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import json
import os
import warnings
from typing import Dict, Any, Optional

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dueling_dqn import DuelingDQN
from double_dueling_dqn import DoubleDuelingDQN
from env_factory import make_env, make_vec_env

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Configuration
# --------------------------------------------------

VARIANT_MAP = {
    "dueling": DuelingDQN,
    "double_dueling": DoubleDuelingDQN,
}

VARIANT_NAMES = {
    "dueling": "Dueling DQN",
    "double_dueling": "Double + Dueling DQN",
}

# Base hyperparameters that are shared across both variants
BASE_HYPERPARAMS = {
    "buffer_size": 100_000,
    "learning_starts": 10_000,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "tau": 1.0,  # Hard update
}

# Variant-specific parameter ranges
VARIANT_SEARCH_SPACES = {
    "dueling": {
        "learning_rate": (1e-5, 1e-3),
        "target_update_interval": [500, 1000, 2000, 5000],
        "batch_size": [32, 64, 128, 256],
        "exploration_fraction": (0.05, 0.5),
        "exploration_initial_eps": (0.5, 1.0),
        "exploration_final_eps": (0.01, 0.1),
        "net_arch": [
            [128, 128],
            [256, 256],
            [512, 512],
            [128, 256, 128],
            [256, 128, 256],
            [256, 512, 256],  # Dueling benefits from deeper networks
        ],
    },
    "double_dueling": {
        "learning_rate": (1e-5, 1e-2),
        "target_update_interval": [500, 1000, 2000, 5000],
        "batch_size": [32, 64, 128, 256],
        "exploration_fraction": (0.05, 0.5),
        "exploration_initial_eps": (0.5, 1.0),
        "exploration_final_eps": (0.01, 0.1),
        "net_arch": [
            [128, 128],
            [256, 256],
            [512, 512],
            [128, 256, 128],
            [256, 128, 256],
            [256, 512, 256],
        ],
    },
}

WARM_START_PARAMS = [
    # Best Double DQN trial (-5143.97) - strongest overall result
    {
        "learning_rate": 0.00010927879883233762,
        "target_update_interval": 1000,
        "batch_size": 256,
        "exploration_fraction": 0.12507445265541792,
        "exploration_initial_eps": 0.9292317317572214,
        "exploration_final_eps": 0.08358632912993018,
        "net_arch": [512, 512],
        "gamma": 0.9654786631566694,
        "learning_starts": 1000,
        "train_freq": 8,
        "gradient_steps": 1,
        "buffer_size": 1000000,
    },
    # Best Vanilla DQN trial (-8474.83)
    {
        "learning_rate": 8.42665034538258e-05,
        "target_update_interval": 1000,
        "batch_size": 128,
        "exploration_fraction": 0.21573775929604358,
        "exploration_initial_eps": 0.9485296583267736,
        "exploration_final_eps": 0.025018757601193223,
        "net_arch": [128, 256, 128],
        "gamma": 0.9907466921420781,
        "learning_starts": 5000,
        "train_freq": 8,
        "gradient_steps": 1,
        "buffer_size": 500000,
    },
    {
        "learning_rate": 6.246073681318086e-05,
        "target_update_interval": 1000,
        "batch_size": 64,
        "exploration_fraction": 0.27163296221848876,
        "exploration_initial_eps": 0.5976214938990223,
        "exploration_final_eps": 0.07502069037353548,
        "net_arch": [256, 128, 256],
        "gamma": 0.9967425002731268,
        "learning_starts": 1000,
        "train_freq": 8,
        "gradient_steps": 1,
        "buffer_size": 1000000,
    },
    {
        "learning_rate": 5.527657643435907e-05,
        "target_update_interval": 2000,
        "batch_size": 128,
        "exploration_fraction": 0.1126944221337712,
        "exploration_initial_eps": 0.6584533293940609,
        "exploration_final_eps": 0.06181433591175487,
        "net_arch": [128, 128],
        "gamma": 0.9756782509256635,
        "learning_starts": 1000,
        "train_freq": 8,
        "gradient_steps": 8,
        "buffer_size": 100000,
    },
]

# Trimmed combos: dropped (1, 8), (1, 16), (4, 16)
TRAIN_FREQ_OPTIONS = [1, 4, 8]
GRADIENT_STEPS_OPTIONS = [1, 4, 8]


def create_objective(
        variant: str,
        eval_freq: int = 50_000,
        total_steps: int = 300_000,
        n_eval_episodes_search: int = 3,
        n_eval_episodes_final: int = 5,
):
    """
    Create objective function for a specific DQN variant.

    Parameters
    ----------
    variant : str
        One of: dueling, double_dueling
    eval_freq : int
        How often to evaluate during training (for pruning)
    total_steps : int
        Total training steps per trial (search budget, not final training)
    n_eval_episodes_search : int
        Episodes used at each pruning checkpoint (kept low for speed)
    n_eval_episodes_final : int
        Episodes used for the final, more thorough evaluation
    """

    AgentClass = VARIANT_MAP[variant]
    search_space = VARIANT_SEARCH_SPACES[variant]

    def objective(trial):
        # Learning rate (log scale)
        lr_min, lr_max = search_space["learning_rate"]
        learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)
        # Target update interval
        target_update_interval = trial.suggest_categorical(
            "target_update_interval",
            search_space["target_update_interval"]
        )
        # Batch size
        batch_size = trial.suggest_categorical(
            "batch_size",
            search_space["batch_size"]
        )
        # Exploration parameters
        exploration_fraction = trial.suggest_float(
            "exploration_fraction",
            search_space["exploration_fraction"][0],
            search_space["exploration_fraction"][1]
        )
        exploration_initial_eps = trial.suggest_float(
            "exploration_initial_eps",
            search_space["exploration_initial_eps"][0],
            search_space["exploration_initial_eps"][1]
        )
        exploration_final_eps = trial.suggest_float(
            "exploration_final_eps",
            search_space["exploration_final_eps"][0],
            search_space["exploration_final_eps"][1]
        )
        # Network architecture
        net_arch = trial.suggest_categorical(
            "net_arch",
            search_space["net_arch"]
        )
        # Additional hyperparameters (common across variants)
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        learning_starts = trial.suggest_categorical(
            "learning_starts",
            [1000, 5000, 10000]
        )
        # Trimmed vs. the vanilla/double script - drops the most
        # expensive train_freq/gradient_steps combos.
        train_freq = trial.suggest_categorical("train_freq", TRAIN_FREQ_OPTIONS)
        gradient_steps = trial.suggest_categorical("gradient_steps", GRADIENT_STEPS_OPTIONS)
        buffer_size = trial.suggest_categorical("buffer_size", [100_000, 500_000, 1_000_000])

        # Training environment (single env for DQN)
        train_env = make_vec_env(
            n_envs=1,
            seed=42,
            use_subprocess=False,
            norm_reward=True
        )
        # Evaluation environment (separate seed)
        eval_env = VecNormalize(
            DummyVecEnv([make_env(rank=100, seed=43)]),
            norm_obs=True,
            norm_reward=False,
            clip_obs=5.0,
            gamma=0.99
        )
        eval_env.training = False

        # Prepare hyperparameters
        hyperparams = {
            **BASE_HYPERPARAMS,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "target_update_interval": target_update_interval,
            "learning_starts": learning_starts,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "exploration_fraction": exploration_fraction,
            "exploration_initial_eps": exploration_initial_eps,
            "exploration_final_eps": exploration_final_eps,
            "gamma": gamma,
            "policy_kwargs": {"net_arch": net_arch},
        }

        # Create model
        model = AgentClass(
            env=train_env,
            update_frequency=4,  # Default, can also be tuned
            tensorboard_log=None,
            device="auto",
            verbose=0,
            **hyperparams,
        )

        n_evaluations = total_steps // eval_freq
        best_reward = -float('inf')

        for step_idx in range(n_evaluations):
            # Train for eval_freq steps
            model.learn(
                total_timesteps=eval_freq,
                reset_num_timesteps=False,
                progress_bar=False
            )

            # Evaluate (fewer episodes than the vanilla/double script)
            mean_reward, _ = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=n_eval_episodes_search,
                deterministic=True
            )

            # Report for pruning
            trial.report(mean_reward, step_idx)

            # Check if trial should be pruned
            if trial.should_prune():
                train_env.close()
                eval_env.close()
                raise optuna.TrialPruned()

            # Track best
            if mean_reward > best_reward:
                best_reward = mean_reward

            # Optional: print progress
            print(f"  Step {step_idx + 1}/{n_evaluations}: Reward = {mean_reward:.2f}")

        # More thorough evaluation at the end (still trimmed vs. 10)
        final_reward, final_std = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes_final,
            deterministic=True
        )

        # Clean up
        train_env.close()
        eval_env.close()

        return final_reward

    return objective



class DuelingDQNOptimizer:
    """Manages optimization for the dueling-family DQN variants."""

    def __init__(
            self,
            n_trials_per_variant: int = 15,
            total_timesteps: int = 300_000,
            eval_freq: int = 50_000,
            n_eval_episodes_search: int = 3,
            n_eval_episodes_final: int = 5,
            output_dir: str = "optimization_results_dueling"
    ):
        self.n_trials_per_variant = n_trials_per_variant
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.n_eval_episodes_search = n_eval_episodes_search
        self.n_eval_episodes_final = n_eval_episodes_final
        self.output_dir = output_dir
        self.results = {}

        os.makedirs(output_dir, exist_ok=True)

    def optimize_variant(self, variant: str) -> Dict[str, Any]:
        print("\n" + "=" * 70)
        print(f"OPTIMIZING: {VARIANT_NAMES[variant]}")
        print("=" * 70)
        print(f"  Trials: {self.n_trials_per_variant}")
        print(f"  Steps per trial: {self.total_timesteps:,}")
        print(f"  Evaluation frequency: {self.eval_freq:,}")
        print("=" * 70 + "\n")

        # Create study
        study_name = f"dqn_{variant}"
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
            interval_steps=1
        )

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=sampler,
            pruner=pruner,
            storage=None  # In-memory storage
        )

        # Warm-start with strong configs found for vanilla/double DQN.
        # skip_if_exists guards against duplicate enqueues if this
        # method is ever called twice on the same study.
        for params in WARM_START_PARAMS:
            study.enqueue_trial(params, skip_if_exists=True)

        # Create objective
        objective = create_objective(
            variant=variant,
            eval_freq=self.eval_freq,
            total_steps=self.total_timesteps,
            n_eval_episodes_search=self.n_eval_episodes_search,
            n_eval_episodes_final=self.n_eval_episodes_final,
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials_per_variant,
            show_progress_bar=True,
            n_jobs=1  # DQN is not thread-safe
        )

        # Extract results
        result = {
            "variant": variant,
            "name": VARIANT_NAMES[variant],
            "best_reward": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "n_complete": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
            "n_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
            "all_trials": [
                {
                    "number": t.number,
                    "value": float(t.value) if t.value is not None else None,
                    "state": str(t.state),
                    "params": t.params if t.state == optuna.trial.TrialState.COMPLETE else None
                }
                for t in study.trials
            ]
        }

        print(f"\n✓ {VARIANT_NAMES[variant]} - Best Reward: {study.best_value:.2f}")
        print(f"  Complete trials: {result['n_complete']}/{result['n_trials']}")
        print(f"  Pruned trials: {result['n_pruned']}")

        return result

    def optimize_all(self):
        """Run optimization for both dueling-family variants."""

        print("=" * 70)
        print("DUELING DQN VARIANTS HYPERPARAMETER OPTIMIZATION")
        print("=" * 70)
        print(f"Total trials across both variants: {len(VARIANT_MAP) * self.n_trials_per_variant}")
        print(f"Total training steps: {len(VARIANT_MAP) * self.n_trials_per_variant * self.total_timesteps:,}")
        print("=" * 70)

        for variant in VARIANT_MAP.keys():
            self.results[variant] = self.optimize_variant(variant)

            # Save after each variant (in case of interruption)
            self.save_results()

    def save_results(self, filename: str = None):
        """Save optimization results to JSON."""
        if filename is None:
            filename = os.path.join(self.output_dir, "dueling_dqn_optimization_results.json")

        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to {filename}")

    def print_summary(self):
        """Print summary of all optimization results."""

        print("\n" + "=" * 70)
        print("OPTIMIZATION SUMMARY")
        print("=" * 70)

        # Sort by best reward
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]["best_reward"],
            reverse=True
        )

        print("\nRank | Variant              | Best Reward | Trials (C/P)")
        print("-" * 60)

        for rank, (variant, data) in enumerate(sorted_results, 1):
            status = f"{data['n_complete']}/{data['n_pruned']}"
            print(f"{rank:4} | {data['name']:20} | {data['best_reward']:10.2f} | {status:>12}")

        print("\n" + "=" * 70)
        print("BEST PARAMETERS PER VARIANT")
        print("=" * 70)

        for variant, data in sorted_results:
            print(f"\n[{data['name']}] Reward: {data['best_reward']:.2f}")
            print("  Parameters:")
            for key, value in data["best_params"].items():
                print(f"    {key:25}: {value}")

    def get_best_variant(self) -> tuple:
        """Return the best performing variant and its results."""
        best_variant = max(
            self.results.keys(),
            key=lambda x: self.results[x]["best_reward"]
        )
        return best_variant, self.results[best_variant]

    def train_best_model(self, total_steps: int = 1_000_000):
        best_variant, best_data = self.get_best_variant()
        AgentClass = VARIANT_MAP[best_variant]

        print("\n" + "=" * 70)
        print(f"TRAINING BEST MODEL: {best_data['name']}")
        print("=" * 70)
        print(f"  Reward: {best_data['best_reward']:.2f}")
        print(f"  Steps: {total_steps:,}")
        print("=" * 70 + "\n")

        # Create environment
        train_env = make_vec_env(
            n_envs=1,
            seed=42,
            use_subprocess=False,
            norm_reward=True
        )

        # Build model with best parameters
        best_params = best_data["best_params"].copy()

        # Ensure required parameters exist
        hyperparams = {
            **BASE_HYPERPARAMS,
            "learning_rate": best_params.get("learning_rate", 1e-4),
            "batch_size": best_params.get("batch_size", 64),
            "buffer_size": best_params.get("buffer_size", 100_000),
            "target_update_interval": best_params.get("target_update_interval", 1000),
            "learning_starts": best_params.get("learning_starts", 10000),
            "train_freq": best_params.get("train_freq", 4),
            "gradient_steps": best_params.get("gradient_steps", 1),
            "exploration_fraction": best_params.get("exploration_fraction", 0.1),
            "exploration_initial_eps": best_params.get("exploration_initial_eps", 1.0),
            "exploration_final_eps": best_params.get("exploration_final_eps", 0.05),
            "gamma": best_params.get("gamma", 0.99),
            "policy_kwargs": {"net_arch": best_params.get("net_arch", [256, 256])},
        }

        model = AgentClass(
            env=train_env,
            update_frequency=4,
            tensorboard_log=None,
            device="auto",
            verbose=1,
            **hyperparams,
        )

        # Train
        model.learn(total_timesteps=total_steps)

        # Save model
        model_path = os.path.join(self.output_dir, f"best_{best_variant}_model")
        model.save(model_path)
        train_env.save(os.path.join(self.output_dir, f"best_{best_variant}_vecnorm.pkl"))

        # Final evaluation
        eval_env = VecNormalize(
            DummyVecEnv([make_env(rank=999, seed=999)]),
            norm_obs=True,
            norm_reward=False,
            clip_obs=5.0,
            gamma=0.99
        )
        eval_env.training = False

        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=50,
            deterministic=True
        )

        print(f"\nFinal Model Performance:")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Model saved to: {model_path}.zip")

        train_env.close()
        eval_env.close()

        return model, mean_reward


# Main
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = DuelingDQNOptimizer(
        n_trials_per_variant=15,   # fewer trials than vanilla/double (was 25)
        total_timesteps=300_000,   # smaller search budget (was 500k)
        eval_freq=50_000,          # same eval frequency
        n_eval_episodes_search=3,  # fewer episodes per checkpoint (was 5)
        n_eval_episodes_final=5,   # fewer episodes at trial end (was 10)
        output_dir="optimization_results_dueling"
    )

    # Run optimization for both dueling variants
    optimizer.optimize_all()

    # Save and print results
    optimizer.save_results()
    optimizer.print_summary()

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {optimizer.output_dir}/")
    print("=" * 70)