from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers._tpe.probability_distributions import _BatchedCategoricalDistributions
from optuna.samplers._tpe.probability_distributions import (
    _BatchedDiscreteTruncLogNormDistributions,
)
from optuna.samplers._tpe.probability_distributions import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedTruncLogNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _MixtureOfProductDistribution


EPS = 1e-12


class _ParzenEstimator:
    def __init__(
        self,
        observations: dict[str, np.ndarray],
        search_space: dict[str, BaseDistribution],
        weights_func: Callable[[int], np.ndarray],
        predetermined_weights: np.ndarray | None = None,
    ) -> None:
        self._search_space = search_space
        transformed_observations = self._transform(observations)

        assert predetermined_weights is None or len(transformed_observations) == len(
            predetermined_weights
        )
        weights = (
            predetermined_weights
            if predetermined_weights is not None
            else weights_func(len(transformed_observations))
        )

        if len(transformed_observations) == 0:
            weights = np.array([1.0])
        else:
            weights = np.append(weights, [1.0])
        weights /= weights.sum()
        self._mixture_distribution = _MixtureOfProductDistribution(
            weights=weights,
            distributions=[
                self._calculate_distributions(transformed_observations[:, i], dist)
                for i, dist in enumerate(search_space.values())
            ],
        )

    def sample(self, rng: np.random.RandomState, size: int) -> dict[str, np.ndarray]:
        sampled = self._mixture_distribution.sample(rng, size)
        return self._untransform(sampled)

    def log_pdf(self, samples_dict: dict[str, np.ndarray]) -> np.ndarray:
        transformed_samples = self._transform(samples_dict)
        return self._mixture_distribution.log_pdf(transformed_samples)

    def _transform(self, samples_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.array([samples_dict[param] for param in self._search_space]).T

    def _untransform(self, samples_array: np.ndarray) -> dict[str, np.ndarray]:
        return {param: samples_array[:, i] for i, param in enumerate(self._search_space)}

    def _calculate_distributions(
        self, observations: np.ndarray, search_space: BaseDistribution
    ) -> _BatchedDistributions:
        if isinstance(search_space, CategoricalDistribution):
            return self._calculate_categorical_distributions(observations, search_space)
        else:
            assert isinstance(search_space, (FloatDistribution, IntDistribution))
            return self._calculate_numerical_distributions(observations, search_space)

    def _calculate_categorical_distributions(
        self, observations: np.ndarray, search_space: CategoricalDistribution
    ) -> _BatchedDistributions:
        choices = search_space.choices
        n_choices = len(choices)
        if len(observations) == 0:
            return _BatchedCategoricalDistributions(
                weights=np.full((1, n_choices), fill_value=1.0 / n_choices)
            )

        n_kernels = len(observations) + 1  # NOTE(sawa3030): +1 for prior.
        weights = np.full(shape=(n_kernels, n_choices), fill_value=1.0 / n_kernels)
        observed_indices = observations.astype(int)
        weights[np.arange(len(observed_indices)), observed_indices] += 1
        row_sums = weights.sum(axis=1, keepdims=True)
        weights /= np.where(row_sums == 0, 1, row_sums)
        return _BatchedCategoricalDistributions(weights)

    def _calculate_numerical_distributions(
        self,
        observations: np.ndarray,
        search_space: FloatDistribution | IntDistribution,
    ) -> _BatchedDistributions:
        low = search_space.low
        high = search_space.high
        if search_space.step is not None:
            low -= search_space.step / 2
            high += search_space.step / 2
        if search_space.log:
            observations = np.log(observations)
            low = np.log(low)
            high = np.log(high)

        mus = observations

        def compute_sigmas() -> np.ndarray:
            sorted_indices = np.argsort(mus)
            sorted_mus = mus[sorted_indices]
            sorted_mus_with_endpoints = np.empty(len(mus) + 2, dtype=float)
            sorted_mus_with_endpoints[0] = low
            sorted_mus_with_endpoints[1:-1] = sorted_mus
            sorted_mus_with_endpoints[-1] = high

            sorted_sigmas = np.maximum(
                sorted_mus_with_endpoints[1:-1] - sorted_mus_with_endpoints[0:-2],
                sorted_mus_with_endpoints[2:] - sorted_mus_with_endpoints[1:-1],
            )

            if sorted_mus_with_endpoints.shape[0] >= 4:
                sorted_sigmas[0] = sorted_mus_with_endpoints[2] - sorted_mus_with_endpoints[1]
                sorted_sigmas[-1] = (
                    sorted_mus_with_endpoints[-2] - sorted_mus_with_endpoints[-3]
                )

            sigmas = sorted_sigmas[np.argsort(sorted_indices)]

            # We adjust the range of the 'sigmas'.
            maxsigma = high - low
            n_kernels = len(observations) + 1  # NOTE(sawa3030): +1 for prior.
            minsigma = (high - low) / min(100.0, (1.0 + n_kernels))
            return np.asarray(np.clip(sigmas, minsigma, maxsigma))

        sigmas = compute_sigmas()
        mus = np.append(mus, [0.5 * (low + high)])
        sigmas = np.append(sigmas, [high - low])

        if search_space.step is None:
            if not search_space.log:
                return _BatchedTruncNormDistributions(
                    mus, sigmas, search_space.low, search_space.high
                )
            else:
                return _BatchedTruncLogNormDistributions(
                    mus, sigmas, search_space.low, search_space.high
                )
        else:
            if not search_space.log:
                return _BatchedDiscreteTruncNormDistributions(
                    mus, sigmas, search_space.low, search_space.high, search_space.step
                )
            else:
                return _BatchedDiscreteTruncLogNormDistributions(
                    mus, sigmas, search_space.low, search_space.high, search_space.step
                )
