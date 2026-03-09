import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark summary")
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.dirname(args.summary)), "figures")
    os.makedirs(outdir, exist_ok=True)

    sns.set_theme(style="whitegrid")

    for level in sorted(summary["level"].unique()):
        sub = summary[summary["level"] == level].copy()

        heat_money = sub.pivot(index="horizon", columns="n_scenarios", values="mean_final_money")
        plt.figure(figsize=(8, 5))
        sns.heatmap(heat_money, annot=True, fmt=".1f", cmap="YlGnBu")
        plt.title(f"{level}: Mean Final Money")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{level}_mean_final_money_heatmap.png"), dpi=150)
        plt.close()

        heat_success = sub.pivot(index="horizon", columns="n_scenarios", values="success_rate")
        plt.figure(figsize=(8, 5))
        sns.heatmap(heat_success, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1)
        plt.title(f"{level}: Success Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{level}_success_rate_heatmap.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=sub,
            x="mean_avg_step_time_ms",
            y="mean_final_money",
            hue="horizon",
            size="n_scenarios",
            palette="viridis",
        )
        plt.title(f"{level}: Pareto (Time vs Money)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{level}_pareto_time_money.png"), dpi=150)
        plt.close()

    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()
