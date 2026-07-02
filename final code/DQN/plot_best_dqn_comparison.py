"""
Reads the three best_model_metadata.json files produced by:
    python select_best_dqn.py --freq 1
    python select_best_dqn.py --freq 4
    python select_best_dqn.py --freq 8

Produces one figure with four side-by-side bar charts:
    - Mean Episode Reward   (higher is better)
    - Infrastructure Cost   (lower is better)
    - Dropped Requests      (lower is better)
    - Queue Occupancy       (lower is better)

Each chart has three bars one per frequency (1, 4, 8).
Error bars show std from the metadata JSON.
The winning variant name is shown in the x-axis label so you
can see which variant won at each frequency.

Usage
-----
    python plot_best_dqn_comparison.py

Output
------
    ./results/plots/best_dqn_frequency_comparison.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

METADATA_PATHS = {
    1: "./models/best_final_dqn_freq1/best_model_metadata.json",
    4: "./models/best_final_dqn_freq4/best_model_metadata.json",
    8: "./models/best_final_dqn_freq8/best_model_metadata.json",
}

OUT_PATH = "./results/plots/best_dqn_frequency_comparison.png"
FREQUENCIES = [1, 4, 8]
BAR_COLORS  = {1: "steelblue", 4: "darkorange", 8: "crimson"}


def load_metadata(freq: int) -> dict:
    path = METADATA_PATHS[freq]
    if not os.path.exists(path):
        print(f"  [MISSING] freq={freq} — {path}")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    os.makedirs("./results/plots/", exist_ok=True)

    records = {}
    for freq in FREQUENCIES:
        data = load_metadata(freq)
        if data is not None:
            records[freq] = data

    if not records:
        print("No metadata files found. Run select_best_dqn.py first.")
        return

    def short_label(variant_name: str) -> str:
        """Shorten long variant names so they fit under the bar."""
        return (variant_name
                .replace("Double + Dueling DQN", "Double+Dueling")
                .replace("Vanilla DQN", "Vanilla")
                .replace("Double DQN", "Double")
                .replace("Dueling DQN", "Dueling"))

    x_labels = []
    for freq in FREQUENCIES:
        if freq in records:
            variant = short_label(records[freq]["selected_variant"])
            x_labels.append(f"freq={freq}\n({variant})")
        else:
            x_labels.append(f"freq={freq}\n(missing)")

    freqs_present = [f for f in FREQUENCIES if f in records]
    x = np.arange(len(freqs_present))

    metrics = [
        {
            "title": "Mean Episode Reward",
            "field": "return_mean",
            "std_field": "return_std",
            "note": "higher is better",
            "color_key": True,   # use per-freq colors
        },
        {
            "title": "Infrastructure Cost",
            "field": "cost_mean",
            "std_field": "cost_std",
            "note": "lower is better",
            "color_key": True,
        },
        {
            "title": "Dropped Requests",
            "field": "dropped_requests_mean",
            "std_field": "dropped_requests_std",
            "note": "lower is better",
            "color_key": True,
        },
        {
            "title": "Queue Occupancy",
            "field": "queue_occ_mean",
            "std_field": "queue_occ_std",
            "note": "lower is better",
            "color_key": True,
        },
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), dpi=150)
    fig.suptitle(
        "Best DQN Model per Update Frequency — Side-by-Side Comparison",
        fontsize=14, fontweight="bold", y=1.02
    )

    for ax, metric in zip(axes, metrics):
        values = []
        errors = []
        colors = []

        for freq in freqs_present:
            rec = records[freq]
            values.append(rec[metric["field"]])
            errors.append(rec[metric["std_field"]])
            colors.append(BAR_COLORS[freq])

        bars = ax.bar(
            x,
            values,
            yerr=errors,
            color=colors,
            alpha=0.85,
            capsize=5,
            width=0.55,
        )

        for bar, val, err in zip(bars, values, errors):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + err + abs(bar.get_height()) * 0.01,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

        ax.set_title(metric["title"], fontsize=11, fontweight="bold")
        ax.set_ylabel(metric["note"], fontsize=8, color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [x_labels[i] for i, f in enumerate(FREQUENCIES) if f in records],
            fontsize=8
        )
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")

    # print summary table to stdout
    print()
    print(f"{'Freq':<8} {'Variant':<22} {'Reward':>10} {'Cost':>10} "
          f"{'Dropped':>10} {'QueueOcc':>10} {'Score':>8}")
    print("-" * 82)
    for freq in freqs_present:
        r = records[freq]
        print(f"  {freq:<6} {short_label(r['selected_variant']):<22} "
              f"{r['return_mean']:>10.1f} "
              f"{r['cost_mean']:>10.1f} "
              f"{r['dropped_requests_mean']:>10.1f} "
              f"{r['queue_occ_mean']:>10.4f} "
              f"{r['overall_score']:>8.4f}")
    print("-" * 82)


if __name__ == "__main__":
    main()
