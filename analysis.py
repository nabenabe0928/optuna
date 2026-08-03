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
    (line,) = ax.plot(steps, m, **plot_kwargs, markevery=20)
    ax.fill_between(steps, m - s, m + s, alpha=0.2, color=color)
    ax.set_xlim(steps[0] - 0.1, steps[-1] + 0.1)
    return line


def main(df: pd.DataFrame, d: int) -> None:
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
    plt.close(fig)


def compute_average_ranks(
    df: pd.DataFrame, d: int, combos: list[tuple[str, str]]
) -> dict[tuple[str, str], np.ndarray]:
    sub_df = df[df["dimension"] == d]
    ranks_per_combo: dict[tuple[str, str], list[np.ndarray]] = {combo: [] for combo in combos}
    for function_id in sub_df["function_id"].unique():
        func_df = sub_df[sub_df["function_id"] == function_id]
        for seed in func_df["seed"].unique():
            seed_df = func_df[func_df["seed"] == seed]
            trajectories = np.stack(
                [
                    np.minimum.accumulate(
                        seed_df[
                            (seed_df["prior_type"] == prior_type)
                            & (seed_df["kernel_type"] == kernel_type)
                        ]["values"].iloc[0]
                    )[N_INIT:]
                    for prior_type, kernel_type in combos
                ]
            )
            # Lower value is better, so rank 1 goes to the smallest value at each step.
            ranks = trajectories.argsort(axis=0).argsort(axis=0) + 1
            for combo, rank in zip(combos, ranks):
                ranks_per_combo[combo].append(rank)
    return {combo: np.mean(ranks, axis=0) for combo, ranks in ranks_per_combo.items()}


def plot_average_rank(df: pd.DataFrame, d: int) -> None:
    combos = [
        (prior_type, kernel_type)
        for prior_type in ["optuna", "hvarfner"]
        for kernel_type in ["rbf", "matern"]
    ]
    avg_ranks = compute_average_ranks(df, d, combos)
    steps = np.arange(N_INIT, N_INIT + next(iter(avg_ranks.values())).size)
    n_functions = df[df["dimension"] == d]["function_id"].nunique()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(which="minor", color="gray", linestyle=":")
    ax.grid(which="major", color="black")
    ax.set_title(f"Average rank over {n_functions} functions (dimension={d})")
    ax.set_xlabel("Number of Evaluations")
    ax.set_ylabel("Average Rank")
    lines = []
    labels = []
    for prior_type, kernel_type in combos:
        color = COLOR_DICT[prior_type]
        plot_kwargs = dict(color=color, ls=LS_DICT[kernel_type], marker=MARKER_DICT[kernel_type])
        (line,) = ax.plot(steps, avg_ranks[(prior_type, kernel_type)], **plot_kwargs, markevery=20)
        lines.append(line)
        labels.append(f"{kernel_type} & {prior_type}")
    ax.set_xlim(steps[0] - 0.1, steps[-1] + 0.1)
    fig.legend(handles=lines, labels=labels, bbox_to_anchor=(0.95, 0.15), ncols=2)
    plt.savefig(f"prior-bench-rank-{d}d.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = get_dataframe()
    print(len(df))
    for d in [2, 5, 10, 20, 40]:
        main(df, d=d)
        plot_average_rank(df, d=d)
