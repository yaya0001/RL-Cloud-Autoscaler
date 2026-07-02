"""
exp5_common.py — shared code for the per-algorithm γ-sweep drivers
This module holds everything that is common across the 7 driver scripts
(exp5_run_ppo.py, exp5_run_recurrent_ppo.py, exp5_run_a2c.py, exp5_run_dqn.py,
exp5_run_double_dqn.py, exp5_run_dueling_dqn.py, exp5_run_dueling_double_dqn.py):
hyperparameters, env builders, train/eval functions, plotting, and the
result-file merge step.

"""

import json
import os
import sys
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baseline_agent import RuleBasedBaseline
from cloud_env import CloudScalingEnv  # noqa: F401
from dueling_dqn_policy import DuelingDQNPolicy
from env_factory import make_env, make_vec_env

if "CloudScaling-v1" not in gym.envs.registry:
    gym.register(id="CloudScaling-v1", entry_point="cloud_env:CloudScalingEnv")

# ── experiment config ──────────────────────────────────────────────────────────
GAMMA_VALUES = [5.0, 10.0, 20.0, 30.0, 50.0]
NOMINAL      = {"alpha": 1.0, "beta": 0.1, "gamma": 50.0, "omega": 5.0}
RESULTS_DIR  = "results/Experments"

# ── hyperparameters ────────────────────────────────────────────────────────────
PPO_KWARGS = dict(
    learning_rate = 1.4377739650640294e-05,
    n_steps       = 4096,
    batch_size    = 64,
    n_epochs      = 5,
    gamma         = 0.9660969058740851,
    gae_lambda    = 0.9175042883882351,
    clip_range    = 0.1291937132179491,
    ent_coef      = 0.012344366981406195,
    vf_coef       = 0.9315525456344808,
    max_grad_norm = 0.48170949108431693,
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)

RECURRENT_PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

A2C_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=512,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)

DQN_KWARGS = dict(
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
)

DOUBLE_DQN_KWARGS = dict(
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
    policy_kwargs=dict(net_arch=[512, 512])
    # Double DQN is always-on in SB3's DQN via the target network —
    # no extra flag needed; this config is identical to DQN_KWARGS,
    # kept separate so it can diverge later if you tune it independently.
)

DUELING_DQN_KWARGS = dict(
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
)

DUELING_DOUBLE_DQN_KWARGS = dict(
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
)

# ── plot styles ────────────────────────────────────────────────────────────────
AGENT_STYLE = {
    "ppo":                {"color": "#2a78d6", "marker": "o", "ls": "-",  "label": "PPO"},
    "recurrent_ppo":      {"color": "#7b4fd4", "marker": "D", "ls": "-",  "label": "Recurrent PPO"},
    "a2c":                {"color": "#f5a623", "marker": "p", "ls": "-",  "label": "A2C"},
    "dqn":                {"color": "#e34948", "marker": "s", "ls": "-",  "label": "DQN"},
    "double_dqn":         {"color": "#c0392b", "marker": "P", "ls": "-",  "label": "Double DQN"},
    "dueling_dqn":        {"color": "#e67e22", "marker": "h", "ls": "-",  "label": "Dueling DQN"},
    "dueling_double_dqn": {"color": "#922b21", "marker": "*", "ls": "-",  "label": "Dueling Double DQN"},
    "baseline":           {"color": "#1baf7a", "marker": "^", "ls": "--", "label": "Rule-based"},
    "random":             {"color": "#888780", "marker": "x", "ls": ":",  "label": "Random"},
}

# ── per-algorithm registry: name → (train_fn, loader_cls, kwargs, policy) ─────
def _agent_spec(name):
    return {
        "ppo":                (PPO,          PPO_KWARGS,                "MlpPolicy"),
        "recurrent_ppo":      (RecurrentPPO, RECURRENT_PPO_KWARGS,      "MlpLstmPolicy"),
        "a2c":                (A2C,          A2C_KWARGS,                "MlpPolicy"),
        "dqn":                (DQN,          DQN_KWARGS,                "MlpPolicy"),
        "double_dqn":         (DQN,          DOUBLE_DQN_KWARGS,         "MlpPolicy"),
        "dueling_dqn":        (DQN,          DUELING_DQN_KWARGS,        DuelingDQNPolicy),
        "dueling_double_dqn": (DQN,          DUELING_DOUBLE_DQN_KWARGS, DuelingDQNPolicy),
    }[name]

ON_POLICY_AGENTS = {"ppo", "recurrent_ppo", "a2c"}

# ── helpers ────────────────────────────────────────────────────────────────────
def reward_weights_for(g: float) -> tuple:
    return (NOMINAL["alpha"], NOMINAL["beta"], g, NOMINAL["omega"])


def model_dir(g: float) -> str:
    return f"./models/exp5/gamma_{int(g)}"


def per_agent_results_path(agent_name: str) -> str:
    return os.path.join(RESULTS_DIR, f"exp5_gamma_sweep_{agent_name}.json")


def _make_train_env_onpolicy(g: float, seed: int, n_envs: int = 8):
    return make_vec_env(n_envs=n_envs, seed=seed, use_subprocess=True,
                        norm_reward=True, reward_weights=reward_weights_for(g))


def _make_train_env_offpolicy(g: float, seed: int):
    return VecNormalize(
        DummyVecEnv([make_env(rank=0, seed=seed,
                              reward_weights=reward_weights_for(g))]),
        norm_obs=True, norm_reward=True, clip_obs=5.0, gamma=0.99)


def _make_eval_env(g: float, seed: int):
    env = VecNormalize(
        DummyVecEnv([make_env(rank=100, seed=seed,
                              reward_weights=reward_weights_for(g))]),
        norm_obs=True, norm_reward=False, clip_obs=5.0, gamma=0.99)
    env.training = False
    return env


def train_agent(agent_name: str, g: float, timesteps: int, device: str):
    """Train one algorithm at one γ value. Returns (best_model_path, vecnorm_path)."""
    cls, kwargs, policy = _agent_spec(agent_name)
    mdir         = model_dir(g)
    best_path    = f"{mdir}/{agent_name}_best"
    vecnorm_path = f"{mdir}/{agent_name}_vecnorm.pkl"

    print(f"\n  [{agent_name}] γ={g}  timesteps={timesteps:,}")
    os.makedirs(best_path, exist_ok=True)

    if agent_name in ON_POLICY_AGENTS:
        train_env = _make_train_env_onpolicy(g, seed=0)
    else:
        train_env = _make_train_env_offpolicy(g, seed=0)
    eval_env = _make_eval_env(g, seed=0)

    model = cls(policy=policy, env=train_env,
                tensorboard_log=f"./logs/exp5/{agent_name}_gamma_{int(g)}/",
                device=device, verbose=0, **kwargs)

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=best_path,
        eval_freq=10_000, n_eval_episodes=5, deterministic=True, verbose=0)

    model.learn(total_timesteps=timesteps, callback=eval_cb,
                reset_num_timesteps=True)

    train_env.save(vecnorm_path)
    train_env.close()
    eval_env.close()
    return f"{best_path}/best_model.zip", vecnorm_path


# ── evaluation ─────────────────────────────────────────────────────────────────
def evaluate(agent, g: float, vecnorm_path, n_episodes: int, seed: int) -> dict:
    rw     = reward_weights_for(g)
    env_fn = make_env(rank=99, seed=seed, reward_weights=rw)
    vec    = DummyVecEnv([env_fn])

    if vecnorm_path and os.path.exists(vecnorm_path):
        vec = VecNormalize.load(vecnorm_path, vec)
        vec.training    = False
        vec.norm_reward = False

    rewards, costs, drops, qocc = [], [], [], []

    for _ in range(n_episodes):
        obs  = vec.reset()
        done = [False]
        R = c = d = q = steps = 0

        lstm_states    = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done[0]:
            if agent == "random":
                action = np.array([vec.action_space.sample()])
            elif isinstance(agent, RecurrentPPO):
                action, lstm_states = agent.predict(
                    obs, state=lstm_states,
                    episode_start=episode_starts, deterministic=True)
                episode_starts = np.array(done)
            elif hasattr(agent, "predict"):
                raw        = obs[0]
                result     = agent.predict(raw, deterministic=True)
                action_val = result[0] if isinstance(result, tuple) else result
                action     = np.atleast_1d(np.array(action_val))

            obs, r, done, info = vec.step(action)
            R += r[0];  c += info[0]["active"]
            d += info[0]["dropped"];  q += info[0]["queue"]
            steps += 1

        rewards.append(R);  costs.append(c)
        drops.append(d);    qocc.append(q / (steps * 500))

    vec.close()
    agg = lambda x: {"mean": float(np.mean(x)), "std": float(np.std(x))}
    return {"reward": agg(rewards), "cost": agg(costs),
            "dropped": agg(drops), "queue_occ": agg(qocc)}


# ── plotting ───────────────────────────────────────────────────────────────────
PLOT_METRICS = [
    ("reward",    "Mean total reward"),
    ("dropped",   "Mean dropped requests"),
    ("cost",      "Mean operational cost"),
    ("queue_occ", "Mean queue occupancy (frac)"),
]


def plot_results(results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    agent_order = ["ppo", "recurrent_ppo", "a2c", "dqn", "double_dqn",
                   "dueling_dqn", "dueling_double_dqn", "baseline", "random"]
    agents = [a for a in agent_order if a in results]

    for metric_key, metric_label in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))

        for agent in agents:
            st   = AGENT_STYLE[agent]
            xs   = sorted(results[agent].keys(), key=float)
            ys   = [results[agent][g][metric_key]["mean"] for g in xs]
            errs = [results[agent][g][metric_key]["std"]  for g in xs]
            xs_f = [float(x) for x in xs]

            ax.plot(xs_f, ys, color=st["color"], marker=st["marker"],
                    ls=st["ls"], lw=2, ms=6, label=st["label"])
            ax.fill_between(xs_f,
                            np.array(ys) - np.array(errs),
                            np.array(ys) + np.array(errs),
                            color=st["color"], alpha=0.10)

        ax.axvline(x=50.0, color="#b4b2a9", ls="--", lw=1.2, label="Nominal γ=50")
        ax.set_xlabel(r"$\gamma$ (drop penalty weight)", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"Exp 5 — Train-time γ sweep: {metric_label}", fontsize=12)
        ax.set_xscale("log")
        ax.set_xticks([float(g) for g in GAMMA_VALUES])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(labelsize=9)
        ax.grid(True, lw=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, ncol=2)

        out = os.path.join(out_dir, f"exp5_gamma_{metric_key}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")


# ── the routine each driver script calls ───────────────────────────────────────
def run_agent_all_gammas(agent_name: str, loader_cls, args):
    """Train + evaluate ONE algorithm across every γ in GAMMA_VALUES, also
    evaluating baseline/random at each γ for a self-contained result file.
    Writes results/Experments/exp5_gamma_sweep_{agent_name}.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    baseline = RuleBasedBaseline(n_max=10, q_max=500)
    t0       = time.perf_counter()
    results  = {}

    for g in GAMMA_VALUES:
        g_key = str(g)
        mdir  = model_dir(g)
        print(f"\n{'='*60}\n  [{agent_name}]  γ = {g}\n{'='*60}")

        best_path    = f"{mdir}/{agent_name}_best/best_model.zip"
        vecnorm_path = f"{mdir}/{agent_name}_vecnorm.pkl"

        if not args.eval_only:
            best_path, vecnorm_path = train_agent(agent_name, g, args.timesteps, args.device)

        agents_eval = {"baseline": baseline, "random": "random"}
        vecnorm_map = {"baseline": None, "random": None}

        if os.path.exists(best_path):
            agents_eval[agent_name] = loader_cls.load(best_path)
            vecnorm_map[agent_name] = vecnorm_path
            print(f"{agent_name} loaded")
        else:
            print(f"{agent_name} not found at {best_path} — skipping")

        print(f"\n  Evaluating ({args.episodes} episodes each) ...")
        for name, agent in agents_eval.items():
            m = evaluate(agent, g, vecnorm_map[name], args.episodes, args.seed)
            results.setdefault(name, {})[g_key] = m
            print(f"    {name:<22}  reward={m['reward']['mean']:>10.1f}  "
                  f"dropped={m['dropped']['mean']:>8.1f}  "
                  f"cost={m['cost']['mean']:>8.1f}")

    out_json = per_agent_results_path(agent_name)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults for {agent_name} saved → {out_json}")
    print(f"Total wall time: {time.perf_counter()-t0:.1f}s "
          f"({(time.perf_counter()-t0)/60:.1f} min)")
    print("    Once all 7 algorithm files finish, run: python exp5_merge.py")


def build_arg_parser(default_timesteps=1_000_000):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=default_timesteps)
    ap.add_argument("--episodes",  type=int, default=10)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--device",    default="cpu", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--eval_only", action="store_true")
    return ap
