from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.samplers._tpe.probability_distributions import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDistributions
from optuna.trial import FrozenTrial


class _ScottParzenEstimator(_ParzenEstimator):
    """1D ParzenEstimator using the bandwidth selection by Scott's rule."""

    def __init__(
        self,
        param_name: str,
        dist: IntDistribution | CategoricalDistribution,
        counts: np.ndarray,
        consider_prior: bool,
        prior_weight: float,
        categorical_distance_func: Callable[[CategoricalChoiceType, CategoricalChoiceType], float]
        | None,
    ):
        if not isinstance(dist, (CategoricalDistribution, IntDistribution)):
            raise ValueError(
                f"Only IntDistribution and CategoricalDistribution are supported, but got {dist}."
            )

        self._n_grids = len(counts)
        self._param_name = param_name
        self._counts = counts.copy()
        cat_dist_func: dict[
            str, Callable[[CategoricalChoiceType, CategoricalChoiceType], float]
        ] = ({} if categorical_distance_func is None else {param_name: categorical_distance_func})
        super().__init__(
            observations={param_name: np.arange(self._n_grids)[counts > 0.0]},
            search_space={param_name: dist},
            parameters=_ParzenEstimatorParameters(
                consider_prior=consider_prior,
                prior_weight=prior_weight,
                consider_magic_clip=False,
                consider_endpoints=False,
                weights=lambda x: np.empty(0),
                multivariate=True,
                categorical_distance_func=cat_dist_func,
            ),
            predetermined_weights=counts[counts > 0.0],
        )

    def _calculate_numerical_distributions(
        self,
        observations: np.ndarray,
        low: float,
        high: float,
        step: float | None,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        # NOTE: low and high are actually `int` in this class.
        # NOTE: The Optuna TPE bandwidth selection is too wide for this analysis.
        assert step is not None and np.isclose(step, 1.0), "MyPy redefinition."

        n_trials = np.sum(self._counts)
        counts_non_zero = self._counts[self._counts > 0]
        weights = counts_non_zero / n_trials
        mus = np.arange(self.n_grids)[self._counts > 0]
        mean_est = mus @ weights
        sigma_est = np.sqrt((mus - mean_est) ** 2 @ counts_non_zero / max(1, n_trials - 1))

        count_cum = np.cumsum(counts_non_zero)
        idx_q25 = np.searchsorted(count_cum, n_trials // 4, side="left")
        idx_q75 = np.searchsorted(count_cum, n_trials * 3 // 4, side="right")
        IQR = mus[min(mus.size - 1, idx_q75)] - mus[idx_q25]

        # Scott's rule by Scott, D.W. (1992),
        # Multivariate Density Estimation: Theory, Practice, and Visualization.
        sigma_est = 1.059 * min(IQR / 1.34, sigma_est) * n_trials ** (-0.2)
        # To avoid numerical errors. 0.5/1.64 means 1.64sigma (=90%) will fit in the target grid.
        sigmas = np.full_like(mus, max(sigma_est, 0.5 / 1.64), dtype=np.float64)
        if parameters.consider_prior:
            mus = np.append(mus, [0.5 * (low + high)])
            sigmas = np.append(sigmas, [1.0 * (high - low + 1)])

        return _BatchedDiscreteTruncNormDistributions(
            mu=mus,
            sigma=sigmas,
            low=0,
            high=self.n_grids - 1,
            step=1,
        )

    @property
    def n_grids(self) -> int:
        return self._n_grids

    def pdf(self, samples: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf({self._param_name: samples}))


def _get_grids_and_grid_indices_of_trials(
    param_name: str,
    dist: IntDistribution | FloatDistribution,
    trials: list[FrozenTrial],
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(dist, FloatDistribution):
        if dist.log:
            grids = np.linspace(np.log(dist.low), np.log(dist.high), n_steps)
            params = np.log([t.params[param_name] for t in trials])
        else:
            grids = np.linspace(dist.low, dist.high, n_steps)
            params = np.asarray([t.params[param_name] for t in trials])
    elif isinstance(dist, IntDistribution):
        if dist.log:
            log_2_n_grids = int(np.ceil(np.log(dist.high - dist.low + 1) / np.log(2)))
            n_steps_in_log_scale = min(log_2_n_grids, n_steps)
            grids = np.linspace(np.log(dist.low), np.log(dist.high), n_steps_in_log_scale)
            params = np.log([t.params[param_name] for t in trials])
        else:
            n_grids = (dist.high + 1 - dist.low) // dist.step
            grids = (
                np.arange(dist.low, dist.high + 1)[:: dist.step]
                if n_grids <= n_steps
                else np.linspace(dist.low, dist.high, n_steps)
            )
            params = np.asarray([t.params[param_name] for t in trials])
    else:
        assert False, "Should not be reached."

    step_size = grids[1] - grids[0]
    # grids[indices[n] - 1] < param - step_size / 2 <= grids[indices[n]]
    indices = np.searchsorted(grids, params - step_size / 2)
    return grids, indices


def _count_numerical_param_in_grid(
    param_name: str,
    dist: IntDistribution | FloatDistribution,
    trials: list[FrozenTrial],
    n_steps: int,
) -> np.ndarray:
    grids, grid_indices_of_trials = _get_grids_and_grid_indices_of_trials(
        param_name,
        dist,
        trials,
        n_steps,
    )
    unique_vals, counts_in_unique = np.unique(grid_indices_of_trials, return_counts=True)
    counts = np.zeros(grids.size, dtype=np.int32)
    counts[unique_vals] += counts_in_unique
    return counts


def _count_categorical_param_in_grid(
    param_name: str,
    dist: CategoricalDistribution,
    trials: list[FrozenTrial],
) -> np.ndarray:
    choice_to_index = {c: i for i, c in enumerate(dist.choices)}
    unique_vals, counts_in_unique = np.unique(
        [choice_to_index[t.params[param_name]] for t in trials],
        return_counts=True,
    )
    counts = np.zeros(len(dist.choices), dtype=np.int32)
    counts[unique_vals] += counts_in_unique
    return counts


def _build_parzen_estimator(
    param_name: str,
    dist: BaseDistribution,
    trials: list[FrozenTrial],
    n_steps: int,
    consider_prior: bool,
    prior_weight: float,
    categorical_distance_func: Callable[[CategoricalChoiceType, CategoricalChoiceType], float]
    | None,
) -> _ScottParzenEstimator:
    rounded_dist: IntDistribution | CategoricalDistribution
    if isinstance(dist, (IntDistribution, FloatDistribution)):
        counts = _count_numerical_param_in_grid(param_name, dist, trials, n_steps)
        rounded_dist = IntDistribution(low=0, high=counts.size - 1)
    elif isinstance(dist, CategoricalDistribution):
        counts = _count_categorical_param_in_grid(param_name, dist, trials)
        rounded_dist = dist
    else:
        raise ValueError(f"Got an unknown dist with the type {type(dist)}.")

    return _ScottParzenEstimator(
        param_name=param_name,
        dist=rounded_dist,
        counts=counts.astype(np.float64),
        consider_prior=consider_prior,
        prior_weight=prior_weight,
        categorical_distance_func=categorical_distance_func,
    )
