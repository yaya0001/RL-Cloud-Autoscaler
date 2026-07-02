"""
exp5_run_dqn.py — γ sweep for DQN only
======================================
Trains and evaluates **DQN** at every γ in [5, 10, 20, 30, 50] and also
evaluates the Rule-based baseline and Random agent at each γ (so this file's
output is a complete, standalone result set for DQN).

Usage
-----
    # full run (all 5 gammas, DQN only)
    python exp5_run_dqn.py

    # smoke test
    python exp5_run_dqn.py --timesteps 20000 --episodes 2

    # skip training, just re-evaluate existing models
    python exp5_run_dqn.py --eval_only

Output
------
results/Experments/exp5_gamma_sweep_dqn.json

Run all 7 exp5_run_*.py files (e.g. one per machine), then combine everything
with:
    python exp5_merge.py
"""

import os
import sys

# Ensure repo root is on sys.path when running from EXP5_dist/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import DQN
from exp5_common import run_agent_all_gammas, build_arg_parser


def main():
    args = build_arg_parser().parse_args()
    run_agent_all_gammas("dqn", DQN, args)


if __name__ == "__main__":
    main()
