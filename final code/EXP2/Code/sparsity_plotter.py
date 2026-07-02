"""
sparsity_plotter.py

Reads evaluations.npz files directly by path

Produces
--------
1) Final performance bar chart at freq=1, all algorithms
2) Final performance bar chart at freq=4, all algorithms
3) Final performance bar chart at freq=8, all algorithms
4) Sample efficiency — timestep each algorithm first crosses reward threshold
5) Stability under sparsity — one unified chart, all algorithms across frequencies

"""

import os
import numpy as np
import matplotlib.pyplot as plt


class SparsityPlotter:
    """Produces sparsity comparison charts from evaluations.npz paths.

    Parameters
    ----------
    eval_paths : dict[(str, int), str]
        Keys are (algorithm_label, frequency) tuples.
        Values are file paths to that run's evaluations.npz.
        Missing or unreadable files are skipped gracefully.
    model_paths : dict[(str, int), tuple]
        Keys are (algorithm_label, frequency) tuples.
        Values are (model_zip_path, vecnorm_pkl_path) tuples.
        Only needed for charts 3 and 4 (scatter plots).
        If not provided, scatter plots are skipped.
    out_dir : str
        Directory where PNG files are saved.
    colors : dict[str, str] or None
        Optional fixed color per algorithm label.
    """

    DEFAULT_PALETTE = [
        "darkorange", "green", "crimson", "purple",
        "royalblue", "mediumorchid", "darkcyan", "saddlebrown",
    ]

    MARKER_MAP = {1: "o", 4: "s", 8: "^"}   # circle / square / triangle per frequency

    def __init__(self, eval_paths: dict, out_dir: str = "./results/Experments/plots_exp2/",
                 colors: dict = None, model_paths: dict = None):
        self.eval_paths  = eval_paths
        self.out_dir     = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.algorithms = sorted({label for (label, freq) in eval_paths})
        self.frequencies = sorted({freq for (label, freq) in eval_paths})

        if colors is not None:
            self.colors = colors
        else:
            self.colors = {
                label: self.DEFAULT_PALETTE[i % len(self.DEFAULT_PALETTE)]
                for i, label in enumerate(self.algorithms)
            }

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _load(self, label: str, freq: int):
        """Load one evaluations.npz. Returns dict or None if missing/unreadable."""
        path = self.eval_paths.get((label, freq))
        if path is None or not os.path.exists(path):
            print(f"  [MISSING] {label} freq={freq} — {path}")
            return None
        try:
            data    = np.load(path)
            steps   = data["timesteps"]
            results = data["results"]
        except Exception as e:
            print(f"  [ERROR] {label} freq={freq} — {path}: {e}")
            return None
        return {"steps": steps, "means": results.mean(axis=1), "stds": results.std(axis=1)}

    @staticmethod
    def _smooth(values, weight=0.9):
        """EMA smoothing."""
        last, out = values[0], []
        for v in values:
            last = last * weight + (1 - weight) * v
            out.append(last)
        return np.array(out)

    # --------------------------------------------------
    # Chart 1-3: Final performance bar charts (one per frequency)
    # --------------------------------------------------

    def plot_final_performance(self, freq: int, filename: str = None):
        """Bar chart of final mean reward, all algorithms, one frequency."""
        labels, finals, errs, colors = [], [], [], []

        for label in self.algorithms:
            curve = self._load(label, freq)
            if curve is None:
                continue
            labels.append(label)
            finals.append(curve["means"][-1])
            errs.append(curve["stds"][-1])
            colors.append(self.colors[label])

        if not labels:
            print(f"  [SKIP] No data for freq={freq}")
            return

        fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
        x = np.arange(len(labels))
        ax.bar(x, finals, yerr=errs, color=colors, alpha=0.85, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Final Mean Episode Reward")
        ax.set_title(f"Final Performance — All Algorithms (update_freq={freq})",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        out_name = filename or f"final_performance_freq{freq}.png"
        plt.savefig(os.path.join(self.out_dir, out_name))
        plt.close(fig)
        print(f"  Saved {out_name}")

    # --------------------------------------------------
    # Chart 4: Sample efficiency
    # --------------------------------------------------

    def plot_sample_efficiency(self, threshold: float = -10000.0, filename: str = None):
        """Bar chart: timestep at which each algorithm × frequency first
        crosses `threshold` reward. Algorithms that never cross it are
        shown at max timesteps with a hatched bar so they're still visible.

        Files needed: evaluations.npz (already in eval_paths).
        """
        # Collect (label, freq, crossing_timestep) triples
        records = []
        for label in self.algorithms:
            for freq in self.frequencies:
                curve = self._load(label, freq)
                if curve is None:
                    continue
                crossed = np.where(curve["means"] >= threshold)[0]
                if len(crossed) > 0:
                    t = int(curve["steps"][crossed[0]])
                else:
                    t = int(curve["steps"][-1])   # never crossed — use max
                records.append((label, freq, t, len(crossed) == 0))

        if not records:
            print("  [SKIP] No data for sample efficiency chart")
            return

        # Group by algorithm, then frequency within each group
        fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
        n_freqs  = len(self.frequencies)
        n_algos  = len(self.algorithms)
        width    = 0.8 / n_freqs
        freq_offsets = {f: (i - n_freqs / 2 + 0.5) * width
                        for i, f in enumerate(self.frequencies)}

        for label, freq, t, never_crossed in records:
            algo_idx = self.algorithms.index(label)
            x        = algo_idx + freq_offsets[freq]
            hatch    = "//" if never_crossed else None
            ax.bar(x, t, width=width * 0.9,
                   color=self.colors[label], alpha=0.75,
                   hatch=hatch, edgecolor="black", linewidth=0.5,
                   label=f"freq={freq}" if algo_idx == 0 else "")

        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xticks(range(n_algos))
        ax.set_xticklabels(self.algorithms, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Timestep of First Crossing")
        ax.set_title(f"Sample Efficiency — First Timestep Reaching Reward ≥ {threshold:.0f}\n"
                     f"(hatched = never reached threshold)",
                     fontsize=13, fontweight="bold")
        ax.legend(title="Update Frequency", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        out_name = filename or "sample_efficiency.png"
        plt.savefig(os.path.join(self.out_dir, out_name))
        plt.close(fig)
        print(f"  Saved {out_name}")

    # --------------------------------------------------
    # Chart 5: Stability under sparsity
    # --------------------------------------------------

    def plot_stability_under_sparsity(self, filename: str = None):
        """One unified line chart: x = update_frequency, y = final reward,
        one line per algorithm with error bars from within-run eval std.
        Directly shows which algorithms degrade under sparse updates.

        Files needed: evaluations.npz (already in eval_paths).
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        plotted_any = False

        for label in self.algorithms:
            means_per_freq, stds_per_freq, freqs_found = [], [], []
            for freq in self.frequencies:
                curve = self._load(label, freq)
                if curve is None:
                    continue
                means_per_freq.append(curve["means"][-1])
                stds_per_freq.append(curve["stds"][-1])
                freqs_found.append(freq)

            if not freqs_found:
                continue

            ax.errorbar(freqs_found, means_per_freq, yerr=stds_per_freq,
                        label=label, color=self.colors[label],
                        linewidth=2, marker="o", markersize=6,
                        capsize=4, capthick=1.5)
            plotted_any = True

        if not plotted_any:
            print("  [SKIP] No data for stability chart")
            plt.close(fig)
            return

        ax.set_xticks(self.frequencies)
        ax.set_xticklabels([f"freq={f}" for f in self.frequencies])
        ax.set_xlabel("Update Frequency")
        ax.set_ylabel("Final Mean Episode Reward")
        ax.set_title("Stability Under Sparsity — All Algorithms Across Update Frequencies",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="lower left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_name = filename or "stability_under_sparsity.png"
        plt.savefig(os.path.join(self.out_dir, out_name))
        plt.close(fig)
        print(f"  Saved {out_name}")

    # --------------------------------------------------
    # Master runner
    # --------------------------------------------------

    def plot_all(self, efficiency_threshold: float = -10000.0):
        """Generate all charts."""
        print("=" * 60)
        print("Generating sparsity comparison charts ...")
        print("=" * 60)

        # Charts 1-3: final performance per frequency
        for freq in self.frequencies:
            self.plot_final_performance(freq)

        # Chart 4: sample efficiency
        self.plot_sample_efficiency(threshold=efficiency_threshold)

        # Chart 5: stability under sparsity
        self.plot_stability_under_sparsity()

        print(f"\nDone — all charts saved to {self.out_dir}")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":

    eval_paths = {
        ("Vanilla DQN", 1): "./logs/vanilla_dqn_freq1_eval/evaluations.npz",
        ("Vanilla DQN", 4): "./logs/vanilla_dqn_freq4_eval/evaluations.npz",
        ("Vanilla DQN", 8): "./logs/vanilla_dqn_freq8_eval/evaluations.npz",
        ("Double DQN", 1): "./logs/double_dqn_freq1_eval/evaluations.npz",
        ("Double DQN", 4): "./logs/double_dqn_freq4_eval/evaluations.npz",
        ("Double DQN", 8): "./logs/double_dqn_freq8_eval/evaluations.npz",
        ("Dueling DQN", 1): "./logs/dueling_dqn_freq1_eval/evaluations.npz",
        ("Dueling DQN", 4): "./logs/dueling_dqn_freq4_eval/evaluations.npz",
        ("Dueling DQN", 8): "./logs/dueling_dqn_freq8_eval/evaluations.npz",
        ("Double+Dueling DQN", 1): "./logs/double_dueling_dqn_freq1_eval/evaluations.npz",
        ("Double+Dueling DQN", 4): "./logs/double_dueling_dqn_freq4_eval/evaluations.npz",
        ("Double+Dueling DQN", 8): "./logs/double_dueling_dqn_freq8_eval/evaluations.npz",
        ("PPO", 1): "./logs/sparse_ppo_k1_eval/evaluations.npz",
        ("PPO", 4): "./logs/sparse_ppo_k4_eval/evaluations.npz",
        ("PPO", 8): "./logs/sparse_ppo_k8_eval/evaluations.npz",
        ("PPO-LSTM", 1): "./logs/sparse_recurrent_ppo_k1_eval/evaluations.npz",
        ("PPO-LSTM", 4): "./logs/sparse_recurrent_ppo_k4_eval/evaluations.npz",
        ("PPO-LSTM", 8): "./logs/sparse_recurrent_ppo_k8_eval/evaluations.npz",
        ("A2C", 1): "./logs/sparse_a2c_k1_eval/evaluations.npz",
        ("A2C", 4): "./logs/sparse_a2c_k4_eval/evaluations.npz",
        ("A2C", 8): "./logs/sparse_a2c_k8_eval/evaluations.npz",
    }

    plotter = SparsityPlotter(
        eval_paths=eval_paths,
        out_dir="./results/Experments/plots_exp2/",
    )
    plotter.plot_all(efficiency_threshold=-10000.0)