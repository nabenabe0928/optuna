from __future__ import annotations

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["text.usetex"] = True

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


def plot_ax(ax: plt.Axes, func_name: str) -> tuple[list[plt.Line2D], list[str]]:
    df = pd.read_json("results.json")
    df_filtered = df[df.func_name == func_name]
    ax.set_title(func_dict[func_name])
    labels = []
    lines = []
    for opt_key, opt_label in opt_dict.items():
        values = np.array(df_filtered[df_filtered.opt_name == opt_key]["values"].to_list())
        cum_min_values = np.minimum.accumulate(values, axis=-1)
        dx = np.arange(cum_min_values.shape[-1]) + 1
        m = np.mean(cum_min_values, axis=0)
        s = np.std(cum_min_values, axis=0) / np.sqrt(cum_min_values.shape[0])
        line, = ax.plot(dx, m, color=color_dict[opt_key], linestyle=linestyle_dict[opt_key])
        ax.fill_between(dx, m - s, m + s, alpha=0.2, color=color_dict[opt_key])
        lines.append(line)
        labels.append(opt_label)
    
    ax.grid()
    ax.set_xlim(1, 100)
    return lines, labels


def plot_figure():
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
            ax.set_ylim(2.0, 17)

        lines, labels = plot_ax(ax, func_name)

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


if __name__ == "__main__":
    plot_figure()
