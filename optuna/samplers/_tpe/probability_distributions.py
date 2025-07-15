from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna.samplers._tpe import _truncnorm


if TYPE_CHECKING:
    from typing import Protocol

    class _BatchedDistributions(Protocol):
        def sample(self, rng: np.random.RandomState, active_indices: np.ndarray) -> np.ndarray: ...

        def log_pdf(self, x: np.ndarray) -> np.ndarray: ...


class _BatchedCategoricalDistributions:
    def __init__(self, weights: np.ndarray):
        self._weights = weights

    def sample(self, rng: np.random.RandomState, active_indices: np.ndarray) -> np.ndarray:
        active_weights = self._weights[active_indices, :]
        rnd_quantile = rng.rand(active_indices.size)
        cum_probs = np.cumsum(active_weights, axis=-1)
        assert np.isclose(cum_probs[:, -1], 1).all()
        cum_probs[:, -1] = 1  # Avoid numerical errors.
        return np.sum(cum_probs < rnd_quantile[:, None], axis=-1)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return np.log(
            np.take_along_axis(
                self._weights[None, :, :], x[:, None, None].astype(np.int64), axis=-1
            )
        )[:, :, 0]


class _BatchedTruncNormDistributions:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, low: float, high: float) -> None:
        self._mu = mu
        self._sigma = sigma
        # Currently, low and high do not change per trial.
        self._low = low
        self._high = high

    def sample(self, rng: np.random.RandomState, active_indices: np.ndarray) -> np.ndarray:
        active_mus = self._mu[active_indices]
        active_sigmas = self._sigma[active_indices]
        return _truncnorm.rvs(
            a=(self._low - active_mus) / active_sigmas,
            b=(self._high - active_mus) / active_sigmas,
            loc=active_mus,
            scale=active_sigmas,
            random_state=rng,
        )

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return _truncnorm.logpdf(
            x=x[:, None],
            a=(self._low - self._mu[None, :]) / self._sigma[None, :],
            b=(self._high - self._mu[None, :]) / self._sigma[None, :],
            loc=self._mu[None, :],
            scale=self._sigma[None, :],
        )


class _BatchedDiscreteTruncNormDistributions:
    def __init__(
        self, mu: np.ndarray, sigma: np.ndarray, low: float, high: float, step: float
    ) -> None:
        self._mu = mu
        self._sigma = sigma
        # Currently, low and high do not change per trial.
        self._low = low
        self._high = high
        self._step = step

    def sample(self, rng: np.random.RandomState, active_indices: np.ndarray) -> np.ndarray:
        active_mus = self._mu[active_indices]
        active_sigmas = self._sigma[active_indices]
        samples = _truncnorm.rvs(
            a=(self._low - self._step / 2 - active_mus) / active_sigmas,
            b=(self._high + self._step / 2 - active_mus) / active_sigmas,
            loc=active_mus,
            scale=active_sigmas,
            random_state=rng,
        )
        return np.clip(
            self._low + np.round((samples - self._low) / self._step) * self._step,
            self._low,
            self._high,
        )

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        lower_limit = self._low - self._step / 2
        upper_limit = self._high + self._step / 2
        x_lower = np.maximum(x - self._step / 2, lower_limit)
        x_upper = np.minimum(x + self._step / 2, upper_limit)
        log_gauss_mass = _truncnorm._log_gauss_mass(
            (x_lower[:, None] - self._mu[None, :]) / self._sigma[None, :],
            (x_upper[:, None] - self._mu[None, :]) / self._sigma[None, :],
        )
        log_p_accept = _truncnorm._log_gauss_mass(
            (self._low - self._step / 2 - self._mu[None, :]) / self._sigma[None, :],
            (self._high + self._step / 2 - self._mu[None, :]) / self._sigma[None, :],
        )
        return log_gauss_mass - log_p_accept


class _MixtureOfProductDistribution:
    def __init__(self, weights: np.ndarray, distributions: list[_BatchedDistributions]) -> None:
        self.weights = weights
        self.distributions = distributions

    def sample(self, rng: np.random.RandomState, batch_size: int) -> np.ndarray:
        active_indices = rng.choice(len(self.weights), p=self.weights, size=batch_size)
        ret = np.empty((batch_size, len(self.distributions)), dtype=np.float64)
        for i, d in enumerate(self.distributions):
            ret[:, i] = d.sample(rng, active_indices)

        return ret

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        batch_size, n_vars = x.shape
        log_pdfs = np.empty((batch_size, len(self.weights), n_vars), dtype=np.float64)
        for i, d in enumerate(self.distributions):
            log_pdfs[:, :, i] = d.log_pdf(x[:, i])
        weighted_log_pdf = np.sum(log_pdfs, axis=-1) + np.log(self.weights[None, :])
        max_ = weighted_log_pdf.max(axis=1)
        # We need to avoid (-inf) - (-inf) when the probability is zero.
        max_[np.isneginf(max_)] = 0
        with np.errstate(divide="ignore"):  # Suppress warning in log(0).
            return np.log(np.exp(weighted_log_pdf - max_[:, None]).sum(axis=1)) + max_
