from __future__ import annotations

import warnings

import numpy as np

from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def _infer_n_constraints(trials: list[FrozenTrial]) -> int:
    # TODO(nabenabe0928): Migrate this to `trial` or `study` once it is ready.
    constraints = [t.system_attrs.get(_CONSTRAINTS_KEY) for t in trials]
    n_constraints = max([len(c) for c in constraints if c is not None], default=0)
    for t, c in zip(trials, constraints):
        if c is None:
            continue

        n_constraints_of_trial = len(c)
        if n_constraints_of_trial < n_constraints:
            raise ValueError(
                "The number of constraints must be consistent during an optimization, but got "
                f"n_constraints={n_constraints_of_trial} at Trial#{t.number}."
            )

    return n_constraints


def _is_trial_feasible(trial: FrozenTrial, n_constraints: int) -> bool:
    if n_constraints == 0:
        return True

    is_hard_constraint_satisfied = trial.state != TrialState.FAIL
    # If `_CONSTRAINTS_KEY` does not exist in a `trial`, consider it as infeasible.
    return is_hard_constraint_satisfied and all(
        c <= 0 for c in trial.system_attrs.get(_CONSTRAINTS_KEY, [1.0])
    )


def _split_trials_and_get_quantiles_for_constraints(
    trials: list[FrozenTrial],
    n_below_min: int,
    n_constraints: int,
) -> tuple[list[list[FrozenTrial]], list[list[FrozenTrial]], list[float]]:
    if n_constraints == 0:
        warnings.warn("No trials with constraint values were found.")
        return [], [], []

    # TODO(nabenabe0928): Define the case when some constraints are nan.
    # The case above happens once we consider constraints for failed trials.
    # For now, the case above does not happen.
    indices = np.arange(len(trials))
    constraints = [t.system_attrs.get(_CONSTRAINTS_KEY) for t in trials]
    indices_for_constraints = np.array([i for i, c in enumerate(constraints) if c is not None])
    # constraint_vals.shape = (n_constraints, len(indices_for_constraints))
    constraint_vals = np.array([constraints[i] for i in indices_for_constraints]).T
    # Find the n_below_min-th minimum value in each constraint.
    thresholds = np.partition(constraint_vals, kth=n_below_min - 1, axis=-1)[:, n_below_min - 1]
    feasible_trials_list, infeasible_trials_list = [], []
    for threshold, constraint_val in zip(thresholds, constraint_vals):
        # Include at least the indices up to the n_below_min-th min constraint value.
        feasible_indices = indices_for_constraints[constraint_val <= max(0, threshold)]
        infeasible_indices = np.setdiff1d(indices, feasible_indices)
        feasible_trials_list.append([trials[idx] for idx in feasible_indices])
        infeasible_trials_list.append([trials[idx] for idx in infeasible_indices])

    n_trials_with_constraints = max(1, indices_for_constraints.size)
    n_feasibles = np.sum(constraint_vals <= thresholds[:, np.newaxis], axis=1)
    quantiles = n_feasibles / n_trials_with_constraints
    return feasible_trials_list, infeasible_trials_list, quantiles.tolist()


def _sample(
    mpes: list[_ParzenEstimator], rng: np.random.RandomState, size: int, sample_ratio: list[float]
) -> dict[str, np.ndarray]:
    denom = sum(sample_ratio)
    if any(ratio < 0 for ratio in sample_ratio) or denom == 0.0:
        raise ValueError(f"sample_ratio must be a list of positive float, but got {sample_ratio}.")
    if len(mpes) != len(sample_ratio):
        raise ValueError(
            "len(sample_ratio) must be identical to n_constraints+1, "
            f"but got len(sample_ratio)={len(sample_ratio)}."
        )
    if len(mpes) == 1:
        return mpes[0].sample(rng=rng, size=size)

    ratios = [r / denom for r in sample_ratio]
    sample_sizes = [int(np.ceil(ratio * size)) if ratio > 0 else 0 for ratio in ratios]
    samples: dict[str, np.ndarray] = {}
    for mpe, sample_size in zip(mpes, sample_sizes):
        if sample_size == 0:
            continue

        new_samples = mpe.sample(rng=rng, size=sample_size)
        if len(samples) == 0:
            samples = new_samples
        else:
            samples = {
                param_name: np.concatenate([samples[param_name], new_samples[param_name]])
                for param_name in samples
            }

    sampled_size = sum(sample_sizes)
    pop_indices = rng.choice(np.arange(sampled_size), size=sampled_size - size, replace=False)
    return {k: np.delete(v, pop_indices) for k, v in samples.items()}
