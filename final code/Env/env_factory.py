"""Vectorized environment factory for PPO/DQN training scripts."""

import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from cloud_env import CloudScalingEnv  # noqa: F401 -- needed for entry_point

if "CloudScaling-v1" not in gym.envs.registry:
    gym.register(id="CloudScaling-v1", entry_point="cloud_env:CloudScalingEnv")


def make_env(rank, seed=0, **env_kwargs):
    """Return a zero-arg callable that creates one seeded env instance."""
    def _init():
        env = gym.make("CloudScaling-v1", **env_kwargs)
        env.reset(seed=seed + rank)
        return env
    return _init


def make_vec_env(n_envs=8, seed=0, use_subprocess=True, norm_reward=True,
                 **env_kwargs):
    """Build a VecNormalize-wrapped vectorized env for SB3 training.

    SubprocVecEnv gives real CPU parallelism (good for PPO's long rollouts).
    DummyVecEnv is sequential but avoids IPC overhead (better for DQN's
    single-env setup or quick tests).
    """
    env_fns = [make_env(i, seed, **env_kwargs) for i in range(n_envs)]

    if use_subprocess:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=norm_reward,
                           clip_obs=5.0, gamma=0.99)
    return vec_env


if __name__ == "__main__":
    print("Building vec env (n_envs=4, SubprocVecEnv) ...")
    vec_env = make_vec_env(n_envs=4, seed=0, use_subprocess=True)

    obs = vec_env.reset()
    print(f"Obs shape: {obs.shape}")
    assert obs.shape == (4, 5)

    for i in range(10):
        actions = [vec_env.action_space.sample() for _ in range(4)]
        obs, rewards, dones, infos = vec_env.step(actions)
        print(f"  step {i+1:2d} | r[0]={rewards[0]:+.3f}  r[1]={rewards[1]:+.3f}")

    vec_env.close()
    print("\n[OK] Vectorization smoke test passed.")
