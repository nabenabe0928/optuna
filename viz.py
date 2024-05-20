from __future__ import annotations

import os

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["text.usetex"] = True

N_TRIALS = 100

func_dict = {
    "cos-500-8": "$\\texttt{EmbeddingCosine}~(C=500, K=8)$",
    "cos-1000-16": "$\\texttt{EmbeddingCosine}~(C=1000, K=16)$",
    "perm-6": "$\\texttt{PermutationShiftL1}~(p=6)$",
    "perm-7": "$\\texttt{PermutationShiftL1}~(p=7)$",
}
opt_dict = {
    "metric-tpe": "Ours",
    "metric-tpe-no-modification": "Ours (No Modification)",
    "tpe": "Original TPE",
    "random": "Random",
}
color_dict = {
    "metric-tpe": "red",
    "metric-tpe-no-modification": "blue",
    "tpe": "olive",
    "random": "black",
}
linestyle_dict = {
    "metric-tpe": "solid",
    "metric-tpe-no-modification": "dashed",
    "tpe": "dotted",
    "random": "dotted",
}


def plot_trajectory(ax: plt.Axes, values: np.ndarray, color: str, linestyle: str, gapcolor: str | None = None) -> plt.Line2D:
    cum_min_values = np.minimum.accumulate(values, axis=-1)
    dx = np.arange(cum_min_values.shape[-1]) + 1
    m = np.mean(cum_min_values, axis=0)
    s = np.std(cum_min_values, axis=0) / np.sqrt(cum_min_values.shape[0])
    line, = ax.plot(dx, m, color=color, linestyle=linestyle, marker="o", markevery=10, markeredgecolor="black", gapcolor=gapcolor)
    ax.fill_between(dx, m - s, m + s, alpha=0.2, color=color)
    return line


def plot_main_ax(ax: plt.Axes, func_name: str) -> tuple[list[plt.Line2D], list[str]]:
    df = pd.read_json("results.json")
    df_filtered = df[df.func_name == func_name]
    ax.set_title(func_dict[func_name])
    labels = []
    lines = []
    for opt_key, opt_label in opt_dict.items():
        values = np.array(df_filtered[df_filtered.opt_name == opt_key]["values"].to_list())
        lines.append(plot_trajectory(ax, values, color=color_dict[opt_key], linestyle=linestyle_dict[opt_key]))
        labels.append(opt_label)
    
    ax.grid()
    ax.set_xlim(1, N_TRIALS)
    return lines, labels


def plot_ablation_ax(ax: plt.Axes, func_name: str) -> list[plt.Line2D]:
    df = pd.read_json("ablation-study.json")
    df_filtered = df[df.func_name == func_name]
    ax.set_title(func_dict[func_name])
    lines = []
    cm = plt.get_cmap("jet")
    for i, log_base in enumerate(range(2, 11)):
        values = np.array(df_filtered[df_filtered.log_base == log_base]["values"].to_list())
        lines.append(plot_trajectory(ax, values, color=cm(i / 9), linestyle="dashed", gapcolor="black"))

    ax.grid()
    ax.set_xlim(1, N_TRIALS)
    return lines


def plot_main_figure():
    fig, axes = plt.subplots(
        ncols=2,
        nrows=2,
        sharex=True,
        figsize=(20, 10),
        gridspec_kw=dict(hspace=0.13, wspace=0.025)
    )
    fig.supxlabel("\# of Evaluations", y=0.03)
    fig.supylabel("Objective Value", x=0.07)

    for i, func_name in enumerate(func_dict):
        r, c = i // 2, i % 2
        ax = axes[r][c]
        if c == 1:
            ax.tick_params(left=False, labelleft=False)
        if r == 0:
            ax.set_ylim(-0.01, 0.17)
        else:
            ax.set_ylim(1.0, 17)

        lines, labels = plot_main_ax(ax, func_name)

    fig.legend(
        handles=lines,
        loc='upper center',
        labels=labels,
        fontsize=24,
        bbox_to_anchor=(0.5, 0.03),
        fancybox=False,
        ncol=len(labels)
    )
    plt.savefig("figs/perf-over-time.pdf", bbox_inches="tight")


def plot_ablation_figure():
    fig, axes = plt.subplots(
        ncols=2,
        nrows=2,
        sharex=True,
        figsize=(20, 10),
        gridspec_kw=dict(hspace=0.13, wspace=0.025)
    )
    zeros = [[0, 0], [0, 0]]
    cb = axes[0, 0].contourf(zeros, zeros, zeros, np.arange(2, 11), cmap=plt.get_cmap("jet"))
    cbar = fig.colorbar(cb, ax=axes.ravel().tolist(), pad=0.025)
    cbar.ax.set_title("$b_d$", y=1.01)
    fig.supxlabel("\# of Evaluations", x=0.44, y=0.025)
    fig.supylabel("Objective Value", x=0.07)

    for i, func_name in enumerate(func_dict):
        r, c = i // 2, i % 2
        ax = axes[r][c]
        if c == 1:
            ax.tick_params(left=False, labelleft=False)
        if r == 0:
            ax.set_ylim(-0.01, 0.17)
        else:
            ax.set_ylim(1.0, 17)

        lines = plot_ablation_ax(ax, func_name)

    plt.savefig("figs/ablation-study.pdf", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs("figs/", exist_ok=True)
    plot_main_figure()
    plot_ablation_figure()
