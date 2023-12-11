from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import IntDistribution
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.probability_distributions import _BatchedCategoricalDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDistributions
from optuna.samplers._tpe.probability_distributions import _MixtureOfProductDistribution


class _EfficientParzenEstimator(_ParzenEstimator):
    """Fast implementation for 1D ParzenEstimator."""

    def __init__(
        self,
        param_name: str,
        dist: IntDistribution | CategoricalDistribution,
        counts: np.ndarray,
        categorical_distance_func: Callable[[CategoricalChoiceType, CategoricalChoiceType], float]
        | None,
    ):
        self._param_name = param_name
        self._search_space = {param_name: dist}
        self._counts = counts.copy()
        self._n_trials = np.sum(self._counts)
        self._n_grids = len(counts)
        self._categorical_distance_func = categorical_distance_func

        if isinstance(dist, CategoricalDistribution):
            distribution = self._calculate_categorical_distributions_efficient()
        elif isinstance(dist, IntDistribution):
            distribution = self._calculate_numerical_distributions_efficient()
        else:
            raise ValueError(
                f"Only IntDistribution and CategoricalDistribution are supported, but got {dist}."
            )

        self._mixture_distribution = _MixtureOfProductDistribution(
            weights=counts[counts > 0] / self._n_trials,
            distributions=[distribution],
        )

    @property
    def n_grids(self) -> int:
        return self._n_grids

    def pdf(self, samples: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf({self._param_name: samples}))

    def _calculate_categorical_distributions_efficient(self) -> _BatchedDistributions:
        distribution = self._search_space[self._param_name]
        assert isinstance(distribution, CategoricalDistribution), "Mypy redefinition."
        choices = distribution.choices
        n_choices = len(choices)
        if n_choices != self._counts.size:
            raise ValueError(
                f"The shape of counts must be n_choices={n_choices}, "
                f"but got {self._counts.size}."
            )

        dist_func = self._categorical_distance_func
        if dist_func is None:
            weights = np.identity(n_choices)[self._counts > 0]
        else:
            used_indices = set([i for i, c in enumerate(self._counts) if c > 0])
            dists = np.array(
                [
                    [dist_func(choices[i], c) for c in choices]
                    for i in range(n_choices)
                    if i in used_indices
                ]
            )
            max_dists = np.max(dists, axis=1)
            coef = np.log(self._n_trials) * np.log(n_choices) / np.log(6)
            weights = np.exp(-((dists / max_dists[:, np.newaxis]) ** 2) * coef)

        # Add 1e-12 to prevent a numerical error.
        weights += 1e-12
        weights /= np.sum(weights, axis=1, keepdims=True)
        return _BatchedCategoricalDistributions(weights=weights)

    def _calculate_numerical_distributions_efficient(self) -> _BatchedDistributions:
        n_trials = self._n_trials
        counts_non_zero = self._counts[self._counts > 0]
        weights = counts_non_zero / n_trials
        values = np.arange(self.n_grids)[self._counts > 0]
        mean_est = values @ weights
        sigma_est = np.sqrt((values - mean_est) ** 2 @ counts_non_zero / max(1, n_trials - 1))

        count_cum = np.cumsum(counts_non_zero)
        idx_q25 = np.searchsorted(count_cum, n_trials // 4, side="left")
        idx_q75 = np.searchsorted(count_cum, n_trials * 3 // 4, side="right")
        IQR = values[min(values.size - 1, idx_q75)] - values[idx_q25]

        # Scott's rule by Scott, D.W. (1992),
        # Multivariate Density Estimation: Theory, Practice, and Visualization.
        sigma_est = 1.059 * min(IQR / 1.34, sigma_est) * n_trials ** (-0.2)
        # To avoid numerical errors. 0.5/1.64 means 1.64sigma (=90%) will fit in the target grid.
        sigma_est = max(sigma_est, 0.5 / 1.64)
        return _BatchedDiscreteTruncNormDistributions(
            mu=values,
            sigma=np.full_like(values, sigma_est, dtype=np.float64),
            low=0,
            high=self.n_grids - 1,
            step=1,
        )
