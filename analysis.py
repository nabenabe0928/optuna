from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import optunahub
import pandas as pd


bbob = optunahub.load_module("benchmarks/bbob")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
N_INIT = 10
COLOR_DICT = {"hvarfner": "darkred", "optuna": "blue"}
LS_DICT = {"rbf": "dotted", "matern": "dashed"}
MARKER_DICT = {"rbf": "s", "matern": "o"}


def get_dataframe() -> pd.DataFrame:
    """
    Schema of dataframe
    - Columns:
        - function_id (int): 1 -- 24. The function id of bbob.
        - dimension (int): 2, 5, 10. The dimension of the search space.
        - seed (int): 0 -- 9. The random seed. We repeat each experiment for statistical purpose.
        - prior_type (str): hvarfner or optuna. What prior type to use.
        - kernel_type (str): RBF or Matern 5/2.
        - values (list[float]): The length is always 200. Smaller is better.
    """
    trials = json.load(open("prior-benchmark.json"))["trials"]
    rows = []
    for t in trials:
        params = t["params"]
        params["values"] = t["user_attrs"]["values"]
        rows.append(params)
    return pd.DataFrame(rows)


def plot_trajectory(
    ax: plt.Axes,
    df: pd.DataFrame,
    prior_type: str,
    kernel_type: str,
    observed_min_value: float,
) -> plt.Line2D:
    cond = (df["prior_type"] == prior_type) & (df["kernel_type"] == kernel_type)
    values = np.minimum.accumulate(df[cond]["values"].to_list(), axis=-1)[:, N_INIT:]
    values -= observed_min_value - 1e-12
    m = values.mean(axis=0)
    s = values.std(axis=0) / np.sqrt(len(values))
    steps = np.arange(len(m)) + N_INIT
    color = COLOR_DICT[prior_type]
    plot_kwargs = dict(color=color, ls=LS_DICT[kernel_type], marker=MARKER_DICT[kernel_type])
    line, = ax.plot(steps, m, **plot_kwargs, markevery=20)
    ax.fill_between(steps, m - s, m + s, alpha=0.2, color=color)
    ax.set_xlim(steps[0] - 0.1, steps[-1] + 0.1)
    return line


def main(d: int) -> None:
    df = get_dataframe()
    ncols = 4
    nrows = 6
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 20), sharex=True)
    for i, ax in enumerate(axes.flatten(), start=1):
        cond = (df["function_id"] == i) & (df["dimension"] == d)
        target_df = df[cond]
        ax.grid(which="minor", color="gray", linestyle=":")
        ax.grid(which="major", color="black")
        ax.set_title(f"function_id: {i}")
        ax.set_yscale("log")
        problem = bbob.Problem(function_id=i, dimension=d)
        lines = []
        labels = []
        for prior_type in ["optuna", "hvarfner"]:
            for kernel_type in ["rbf", "matern"]:
                line = plot_trajectory(
                    ax,
                    target_df,
                    prior_type=prior_type,
                    kernel_type=kernel_type,
                    observed_min_value=np.min(target_df["values"].to_list()),
                )
                lines.append(line)
                labels.append(f"{kernel_type} & {prior_type}")
    fig.legend(handles=lines, labels=labels, bbox_to_anchor=(0.75, 0.08), ncols=4)
    plt.savefig(f"prior-bench-{d}d.png", bbox_inches="tight")


if __name__ == "__main__":
    main(d=2)
    main(d=5)
    main(d=10)
