from __future__ import annotations

import numpy as np

from analysis import N_INIT
from analysis import compute_average_ranks
from analysis import get_dataframe
from analysis import plt


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
COLOR_DICT = {"hvarfner": "darkred", "optuna": "blue"}
LS_DICT = {"rbf": "dotted", "matern": "dashed"}
MARKER_DICT = {"rbf": "s", "matern": "o"}
DIMENSIONS = [5, 10, 20, 40]


def plot_average_rank_on_ax(ax: plt.Axes, df, d: int) -> tuple[list, list]:
    combos = [
        (prior_type, kernel_type)
        for prior_type in ["optuna", "hvarfner"]
        for kernel_type in ["matern", "rbf"]
    ]
    avg_ranks = compute_average_ranks(df, d, combos)
    steps = np.arange(N_INIT, N_INIT + next(iter(avg_ranks.values())).size)

    ax.grid(which="minor", color="gray", linestyle=":")
    ax.grid(which="major", color="black")
    ax.set_title(f"{d}D")
    lines = []
    labels = []
    for prior_type, kernel_type in combos:
        color = COLOR_DICT[prior_type]
        plot_kwargs = dict(color=color, ls=LS_DICT[kernel_type], marker=MARKER_DICT[kernel_type])
        (line,) = ax.plot(steps, avg_ranks[(prior_type, kernel_type)], **plot_kwargs, ms=10, lw=3, markevery=20)
        lines.append(line)
        prior_label = {"optuna": "Optuna", "hvarfner": "Hvarfner"}[prior_type]
        kernel_label = {"rbf": "RBF", "matern": "Matern 5/2"}[kernel_type]
        labels.append(f"{kernel_label} & {prior_label}")
    ax.set_xlim(steps[0] - 0.1, steps[-1] + 0.1)
    return lines, labels


def main() -> None:
    df = get_dataframe()
    fig, axes = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=(16, 8),
        sharex=True,
        sharey=True,
        gridspec_kw=dict(wspace=0.03, hspace=0.14)
    )
    lines: list = []
    labels: list = []
    for ax, d in zip(axes.flatten(), DIMENSIONS):
        ax.set_ylim(1, 4)
        lines, labels = plot_average_rank_on_ax(ax, df, d)
    fig.legend(
        handles=lines,
        labels=labels,
        bbox_to_anchor=(0.8, 0.02),
        ncols=2,
        fontsize=24,
    )
    fig.supxlabel("Number of Trials", y=0.02)
    fig.supylabel("Average Rank", x=0.08)
    plt.savefig("log-prior-bench-rank.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
