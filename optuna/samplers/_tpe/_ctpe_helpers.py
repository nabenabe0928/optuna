from __future__ import annotations

import warnings

import numpy as np

from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def _infer_n_constraints(trials: list[FrozenTrial]) -> int:
    # TODO(nabenabe0928): Migrate this to `trial` or `study` once it is ready.
    # TODO(nabenabe0928): Define the value of undefined constraints (None? or np.nan?).
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
    n_trials_with_constraints = max(1, indices_for_constraints.size)
    feasible_trials_list, infeasible_trials_list, quantiles = [], [], []
    for threshold, constraint_val in zip(thresholds, constraint_vals):
        # Include at least the indices up to the n_below_min-th min constraint value.
        feasible_indices = indices_for_constraints[constraint_val <= max(0, threshold)]
        infeasible_indices = np.setdiff1d(indices, feasible_indices)
        if infeasible_indices.size == 0:
            # Skip constraints with all feasible, because it does not affect acq_func.
            continue

        feasible_trials_list.append([trials[idx] for idx in feasible_indices])
        infeasible_trials_list.append([trials[idx] for idx in infeasible_indices])
        quantiles.append(np.sum(constraint_val <= 0) / n_trials_with_constraints)

    return feasible_trials_list, infeasible_trials_list, quantiles


def _compute_ctpe_acquisition_func(
    samples: dict[str, np.ndarray],
    mpes_below: list[_ParzenEstimator],
    mpes_above: list[_ParzenEstimator],
    quantiles: list[float],
) -> np.ndarray:
    # See: c-TPE: Tree-structured Parzen Estimator with Inequality Constraints for
    # Expensive Hyperparameter Optimization (https://arxiv.org/abs/2211.14411)
    # NOTE: If no constraint exists, acq_func_vals falls back to the original TPE version.
    # NOTE: Mathematically speaking, the original TPE can also use this acquisition function.
    # NOTE: When removing experimental, we can use the relative density ratio below.
    # TODO(nabenabe0928): Check the reproducibility.
    _quantiles = np.asarray(quantiles)[:, np.newaxis]
    log_first_term = np.log(_quantiles + EPS)
    log_second_term = (
        np.log(1.0 - _quantiles + EPS) + log_likelihoods_above - log_likelihoods_below
    )
    acq_func_vals = np.sum(-np.logaddexp(log_first_term, log_second_term), axis=0)
    return acq_func_vals


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
