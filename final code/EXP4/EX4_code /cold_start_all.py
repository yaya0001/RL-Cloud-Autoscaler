"""
Experiment 4 — Cold-Start Test
===============================
Sweeps boot_delay in {0, 1, 3, 5, 10} and evaluates:
  - PPO
  - Recurrent PPO
  - A2C
  - DQN
  - Double DQN
  - Dueling DQN
  - Dueling Double DQN
  - Rule-based baseline
  - Random agent

Usage:
    python exp4_cold_start.py [--episodes 10] [--ep_len 1000]
"""

import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cloud_env import CloudScalingEnv  # noqa: F401
from env_factory import make_env
from baseline_agent import RuleBasedBaseline

if "CloudScaling-v1" not in gym.envs.registry:
    gym.register(id="CloudScaling-v1", entry_point="cloud_env:CloudScalingEnv")

BOOT_DELAYS = [0, 1, 3, 5, 10]

# agent name → (loader class, model path, vecnorm path)
AGENT_REGISTRY = {
    "ppo":              (PPO,          "models/sparse_ppo_k1.zip",              "models/vecnormalize_sparse_ppo_k1.pkl"),
    "recurrent_ppo":    (RecurrentPPO, "models/sparse_recurrent_ppo_k4.zip",    "models/vecnormalize_sparse_recurrent_ppo_k4.pkl"),
    "a2c":              (A2C,          "models/sparse_a2c_k1.zip",              "models/vecnormalize_sparse_a2c_k1.pkl"),
    "dqn":              (DQN,          "models/best_vanilla_dqn_freq8/best_model.zip",              "models/vecnormalize_vanilla_dqn_freq8.pkl"),
    "double_dqn":       (DQN,          "models/best_double_dqn_freq8/best_model.zip",   "./models/vecnormalize_double_dqn_freq8.pkl"),
    "dueling_dqn":      (DQN,          "models/best_dueling_dqn_freq4/best_model.zip",      "./models/vecnormalize_dueling_dqn_freq4.pkl"),
    "dueling_double_dqn": (DQN,        "models/best_double_dueling_dqn_freq8/best_model.zip", "./models/vecnormalize_double_dueling_dqn_freq8.pkl"),
}


#evaluation

def _make_eval_env(boot_delay: int, seed: int, vecnorm_path: str | None = None):
    env_fn = make_env(rank=99, seed=seed, boot_delay=boot_delay)
    vec = DummyVecEnv([env_fn])
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec = VecNormalize.load(vecnorm_path, vec)
        vec.training = False
        vec.norm_reward = False
    return vec


def evaluate(agent, boot_delay: int, n_episodes: int, seed: int,
             vecnorm_path: str | None = None, ep_len: int = 1000):
    env = _make_eval_env(boot_delay, seed, vecnorm_path)
    rewards, costs, drops, qocc = [], [], [], []

    for _ in range(n_episodes):
        obs = env.reset()
        done = [False]
        R = c = d = q = steps = 0

        # RecurrentPPO needs lstm state carried across steps
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done[0]:
            if agent == "random":
                action = np.array([env.action_space.sample()])
            elif isinstance(agent, RecurrentPPO):
                action, lstm_states = agent.predict(
                    obs, state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True
                )
                episode_starts = np.array(done)
            elif hasattr(agent, "predict"):
                action, _ = agent.predict(obs, deterministic=True)
            else:
                raw_obs = obs[0]
                action_scalar, _ = agent.predict(raw_obs, deterministic=True)
                action = np.array([action_scalar])

            obs, r, done, info = env.step(action)
            R += r[0];  c += info[0]["active"]
            d += info[0]["dropped"];  q += info[0]["queue"]
            steps += 1

        rewards.append(R);  costs.append(c)
        drops.append(d);    qocc.append(q / (steps * 500))

    env.close()

    def agg(x):
        return {"mean": float(np.mean(x)), "std": float(np.std(x))}

    return {"reward": agg(rewards), "cost": agg(costs),
            "dropped": agg(drops), "queue_occ": agg(qocc)}


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--ep_len",   type=int, default=1000)
    args = ap.parse_args()

    #load all models
    agents = {
        "baseline": RuleBasedBaseline(n_max=10, q_max=500),
        "random":   "random",
    }
    vecnorm_map = {"baseline": None, "random": None}

    for name, (cls, model_path, vecnorm_path) in AGENT_REGISTRY.items():
        if os.path.exists(model_path):
            agents[name] = cls.load(model_path)
            vecnorm_map[name] = vecnorm_path
            print(f"Loaded {name}")
        else:
            print(f"{name} not found at {model_path} — skipping")

    #run sweep 
    results = {}

    header = (f"{'agent':<22} {'boot_delay':>10} {'reward':>12} "
              f"{'dropped':>10} {'cost':>10} {'queue_occ':>12}")
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for agent_name, agent in agents.items():
        results[agent_name] = {}
        for bd in BOOT_DELAYS:
            t0 = time.perf_counter()
            m = evaluate(agent, bd, args.episodes, args.seed,
                         vecnorm_path=vecnorm_map[agent_name],
                         ep_len=args.ep_len)
            elapsed = time.perf_counter() - t0
            results[agent_name][bd] = m
            print(
                f"{agent_name:<22} {bd:>10d} "
                f"{m['reward']['mean']:>12.1f} "
                f"{m['dropped']['mean']:>10.1f} "
                f"{m['cost']['mean']:>10.1f} "
                f"{m['queue_occ']['mean']:>12.4f}"
                f"   [{elapsed:.1f}s]"
            )

    print(sep + "\n")

    os.makedirs("results", exist_ok=True)
    out = "results/Experments/exp4_cold_start.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()