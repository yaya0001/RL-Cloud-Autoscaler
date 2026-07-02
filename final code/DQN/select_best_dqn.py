import os
import json
import shutil
import argparse

from stable_baselines3 import DQN

from eval_agent import evaluate_agent
from vanilla_dqn        import VanillaDQN
from double_dqn         import DoubleDQN
from dueling_dqn        import DuelingDQN
from double_dueling_dqn import DoubleDuelingDQN

# Policy classes needed for custom_objects when loading Dueling variants
from dueling_dqn         import DuelingDQNPolicy
from double_dueling_dqn  import DoubleDuelingDQNPolicy

VARIANTS = [
    (VanillaDQN, None),
    (DoubleDQN, None),
    (DuelingDQN, DuelingDQNPolicy),
    (DoubleDuelingDQN, DoubleDuelingDQNPolicy),
]


def compute_overall_score(metrics: dict) -> float:
    """Combine reward, cost, dropped requests, and queue occupancy into
    one comparable score.
    """
    reward_mean,_ = metrics["reward"]
    cost_mean,_ = metrics["cost"]
    dropped_mean,_ = metrics["dropped"]
    qocc_mean,_ = metrics["queue_occ"]

    # Reward reference range observed across your existing eval runs.
    REWARD_FLOOR = -10000.0
    REWARD_CEIL = -2000.0
    reward_norm = (reward_mean - REWARD_FLOOR) / (REWARD_CEIL - REWARD_FLOOR)
    reward_norm = max(0.0, min(1.0, reward_norm))

    # Penalty terms scaled down so reward stays the dominant factor,
    dropped_penalty = min(1.0, dropped_mean / 200.0)   # 200 drops = full penalty
    qocc_penalty = qocc_mean                        # already 0-1 fractional

    overall_score = reward_norm - 0.5 * dropped_penalty - 0.2 * qocc_penalty
    return overall_score


def load_variant_model(AgentClass, custom_policy, freq: int):
    """Load one variant's best_model.zip + vecnorm path.

    Returns (model, vecnorm_path) or (None, None) if files are missing.
    """
    paths = AgentClass.get_paths(freq)
    model_path = os.path.join(paths["best_model"], "best_model.zip")
    vecnorm = paths["vecnorm"]

    if not os.path.exists(model_path):
        print(f"  [SKIP] {AgentClass.LABEL}: model not found at {model_path}")
        return None, None

    if custom_policy is None:
        model = DQN.load(model_path)
    else:
        model = DQN.load(model_path, custom_objects={"policy_class": custom_policy})

    return model, vecnorm


def main():
    parser = argparse.ArgumentParser(
        description="Select the single best DQN variant by overall_score")
    parser.add_argument("--freq", type=int, default=4, choices=[1, 2, 4, 8],
                        help="Which update_frequency's best_model to compare")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Evaluation episodes per variant")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  SELECT BEST DQN — freq={args.freq}")
    print("=" * 60)

    candidates = []   # list of dicts, one per evaluated variant

    for AgentClass, custom_policy in VARIANTS:
        print(f"\nEvaluating {AgentClass.LABEL} ...")
        model, vecnorm = load_variant_model(AgentClass, custom_policy, args.freq)

        if model is None:
            continue

        metrics = evaluate_agent(model, vecnorm, args.episodes, args.seed)
        overall_score = compute_overall_score(metrics)

        paths = AgentClass.get_paths(args.freq)
        candidates.append({
            "label":         AgentClass.LABEL,
            "slug":          AgentClass.SLUG,
            "original_model_path":   os.path.join(paths["best_model"], "best_model.zip"),
            "original_vecnorm_path": paths["vecnorm"],
            "return_mean":   metrics["reward"][0],
            "return_std":    metrics["reward"][1],
            "cost_mean":     metrics["cost"][0],
            "cost_std":      metrics["cost"][1],
            "dropped_requests_mean": metrics["dropped"][0],
            "dropped_requests_std":  metrics["dropped"][1],
            "queue_occ_mean": metrics["queue_occ"][0],
            "queue_occ_std":  metrics["queue_occ"][1],
            "overall_score": overall_score,
        })

        print(f"  reward={metrics['reward'][0]:.1f}  "
              f"dropped={metrics['dropped'][0]:.1f}  "
              f"overall_score={overall_score:.4f}")

    if not candidates:
        print("\nNo variants found to compare. Train at least one first.")
        return

    # Pick the winner
    winner = max(candidates, key=lambda c: c["overall_score"])

    print()
    print("=" * 60)
    print(f"  WINNER: {winner['label']}  "
          f"(overall_score={winner['overall_score']:.4f})")
    print("=" * 60)

    # copy winning model + vecnorm into best_final_dqn/
    out_dir = f"./models/best_final_dqn_freq{args.freq}/"
    os.makedirs(out_dir, exist_ok=True)

    saved_model_path   = os.path.join(out_dir, "best_model.zip")
    saved_vecnorm_path = os.path.join(out_dir, "vecnormalize.pkl")

    shutil.copy2(winner["original_model_path"],   saved_model_path)
    shutil.copy2(winner["original_vecnorm_path"], saved_vecnorm_path)

    # write metadata JSON — same structure as A2C's convention
    metadata = {
        "selected_variant": winner["label"],
        "family": "DQN",
        "selection_rule": "best_overall_score_within_same_family",
        "update_frequency": args.freq,
        "original_model_path": winner["original_model_path"],
        "original_vecnorm_path": winner["original_vecnorm_path"],
        "saved_model_path": saved_model_path,
        "saved_vecnorm_path": saved_vecnorm_path,
        "return_mean": winner["return_mean"],
        "return_std": winner["return_std"],
        "cost_mean": winner["cost_mean"],
        "cost_std": winner["cost_std"],
        "dropped_requests_mean": winner["dropped_requests_mean"],
        "dropped_requests_std": winner["dropped_requests_std"],
        "queue_occ_mean": winner["queue_occ_mean"],
        "queue_occ_std": winner["queue_occ_std"],
        "overall_score": winner["overall_score"],
        "all_candidates": candidates,
    }

    metadata_path = os.path.join(out_dir, "best_model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved winning model to : {saved_model_path}")
    print(f"Saved vecnormalize to : {saved_vecnorm_path}")
    print(f"Saved metadata to : {metadata_path}")


if __name__ == "__main__":
    main()