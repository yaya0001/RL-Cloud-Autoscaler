"""
exp5_merge.py — combine the 7 per-algorithm result files
Run this after all 7 exp5_run_*.py driver scripts have finished (on however
many machines you split them across). Copy every

    results/Experments/exp5_gamma_sweep_{agent}.json

file into one results/Experments/ folder, then run:

    python exp5_merge.py

This writes the combined results/Experments/exp5_gamma_sweep.json (same
shape as the original single-process script) and regenerates
results/Experments/plots_ex5/exp5_gamma_*.png.
"""

import argparse
import glob
import json
import os
import sys

# Ensure repo root is on sys.path when running from EXP5_dist/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exp5_common import RESULTS_DIR, GAMMA_VALUES, plot_results

EXPECTED_AGENTS = [
    "ppo", "recurrent_ppo", "a2c", "dqn",
    "double_dqn", "dueling_dqn", "dueling_double_dqn",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/Experments/plots_ex5")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "exp5_gamma_sweep_*.json")))
    files = [f for f in files if not f.endswith("exp5_gamma_sweep.json")]

    if not files:
        raise SystemExit(
            f"No per-algorithm result files found in {RESULTS_DIR}/ "
            f"(expected exp5_gamma_sweep_{{agent}}.json). "
            f"Copy the JSON output from each of the 7 exp5_run_*.py runs "
            f"into this folder first."
        )

    print(f"Merging {len(files)} result file(s):")
    merged = {}
    for fp in files:
        print(f"  + {fp}")
        with open(fp) as f:
            part = json.load(f)
        for agent_name, by_gamma in part.items():
            merged.setdefault(agent_name, {}).update(by_gamma)

    found_agents = set(merged.keys()) - {"baseline", "random"}
    missing_agents = set(EXPECTED_AGENTS) - found_agents
    if missing_agents:
        print(f" Warning: missing algorithm result files for: {sorted(missing_agents)}")

    for agent_name, by_gamma in merged.items():
        missing_g = {str(g) for g in GAMMA_VALUES} - set(by_gamma.keys())
        if missing_g:
            print(f" Warning: {agent_name} is missing γ values: {sorted(missing_g, key=float)}")

    out_json = os.path.join(RESULTS_DIR, "exp5_gamma_sweep.json")
    with open(out_json, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged results saved → {out_json}")

    print("\nGenerating plots ...")
    plot_results(merged, args.out_dir)


if __name__ == "__main__":
    main()
