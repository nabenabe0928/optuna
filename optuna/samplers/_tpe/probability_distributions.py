from __future__ import annotations

from typing import NamedTuple
from typing import Union

import numpy as np

from optuna.samplers._tpe import _truncnorm


class _BatchedCategoricalDistributions(NamedTuple):
    weights: np.ndarray


class _BatchedTruncNormDistributions(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    low: float  # Currently, low and high do not change per trial.
    high: float


class _BatchedDiscreteTruncNormDistributions(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    low: float  # Currently, low, high and step do not change per trial.
    high: float
    step: float


_BatchedDistributions = Union[
    _BatchedCategoricalDistributions,
    _BatchedTruncNormDistributions,
    _BatchedDiscreteTruncNormDistributions,
]


def _sample_numerical_params(
    dists: list[_BatchedTruncNormDistributions | _BatchedDiscreteTruncNormDistributions],
    rng: np.random.RandomState,
    active_indices: np.ndarray,
) -> np.ndarray:
    lows = np.array([d.low for d in dists])
    highs = np.array([d.high for d in dists])
    steps = np.array(
        [d.step if isinstance(d, _BatchedDiscreteTruncNormDistributions) else 0.0 for d in dists]
    )
    mus = np.array([d.mu[active_indices] for d in dists]).T
    sigmas = np.array([d.sigma[active_indices] for d in dists]).T
    samples = _truncnorm.rvs(
        a=(lows - steps / 2 - mus) / sigmas,
        b=(highs + steps / 2 - mus) / sigmas,
        loc=mus,
        scale=sigmas,
        random_state=rng,
    )
    is_disc = steps != 0.0
    padded_steps = np.where(is_disc, steps, 1.0)  # Pad to prevent zero division.
    return np.clip(
        np.where(is_disc, lows + np.round((samples - lows) / padded_steps) * steps, samples),
        lows,
        highs,
    )


def _log_pdf_continuous(x: np.ndarray, dists: list[_BatchedTruncNormDistributions]) -> np.ndarray:
    # x.shape = (batch_size, dim).
    lows = np.array([d.low for d in dists])
    highs = np.array([d.high for d in dists])
    # {mus, sigmas}.shape = (1, n_kernels, dim).
    mus = np.array([d.mu for d in dists]).T[np.newaxis]
    sigmas = np.array([d.sigma for d in dists]).T[np.newaxis]
    return _truncnorm.logpdf(
        x=x[:, np.newaxis],
        a=(lows - mus) / sigmas,
        b=(highs - mus) / sigmas,
        loc=mus,
        scale=sigmas,
    )


def _log_pdf_discrete(
    x: np.ndarray, dists: list[_BatchedDiscreteTruncNormDistributions]
) -> np.ndarray:
    # x.shape = (batch_size, dim).
    lows = np.array([d.low for d in dists])
    highs = np.array([d.high for d in dists])
    half_steps = np.array([d.step for d in dists]) / 2
    # {mus, sigmas}.shape = (1, n_kernels, dim).
    mus = np.array([d.mu for d in dists]).T[np.newaxis]
    sigmas = np.array([d.sigma for d in dists]).T[np.newaxis]
    lower_limit = lows - half_steps
    upper_limit = highs + half_steps
    x_lower = np.maximum(x - half_steps, lower_limit)[:, np.newaxis]
    x_upper = np.minimum(x + half_steps, upper_limit)[:, np.newaxis]
    # results.shape = (batch_size, n_kernels, dim).
    log_gauss_mass = _truncnorm._log_gauss_mass((x_lower - mus) / sigmas, (x_upper - mus) / sigmas)
    log_p_accept = _truncnorm._log_gauss_mass(
        (lower_limit - mus) / sigmas, (upper_limit - mus) / sigmas
    )
    return log_gauss_mass - log_p_accept


class _MixtureOfProductDistribution(NamedTuple):
    weights: np.ndarray
    distributions: list[_BatchedDistributions]

    def sample(self, rng: np.random.RandomState, batch_size: int) -> np.ndarray:
        active_indices = rng.choice(len(self.weights), p=self.weights, size=batch_size)

        ret = np.empty((batch_size, len(self.distributions)), dtype=np.float64)
        numerical_dists = []
        numerical_indices = []
        for i, d in enumerate(self.distributions):
            if isinstance(d, _BatchedCategoricalDistributions):
                active_weights = d.weights[active_indices, :]
                rnd_quantile = rng.rand(batch_size)
                cum_probs = np.cumsum(active_weights, axis=-1)
                assert np.isclose(cum_probs[:, -1], 1).all()
                cum_probs[:, -1] = 1  # Avoid numerical errors.
                ret[:, i] = np.sum(cum_probs < rnd_quantile[:, None], axis=-1)
            elif isinstance(
                d, (_BatchedTruncNormDistributions, _BatchedDiscreteTruncNormDistributions)
            ):
                numerical_dists.append(d)
                numerical_indices.append(i)
            else:
                assert False

        ret[:, numerical_indices] = _sample_numerical_params(numerical_dists, rng, active_indices)
        return ret

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        batch_size, n_vars = x.shape
        log_pdfs = np.empty((batch_size, len(self.weights), n_vars), dtype=np.float64)
        continuous_dists = []
        continuous_indices = []
        discrete_dists = []
        discrete_indices = []
        for i, d in enumerate(self.distributions):
            xi = x[:, i]
            if isinstance(d, _BatchedCategoricalDistributions):
                log_pdfs[:, :, i] = np.log(
                    np.take_along_axis(
                        d.weights[None, :, :], xi[:, None, None].astype(np.int64), axis=-1
                    )
                )[:, :, 0]
            elif isinstance(d, _BatchedTruncNormDistributions):
                continuous_dists.append(d)
                continuous_indices.append(i)
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                discrete_dists.append(d)
                discrete_indices.append(i)
            else:
                assert False

        log_pdfs[..., continuous_indices] = _log_pdf_continuous(
            x[:, continuous_indices], continuous_dists
        )
        log_pdfs[..., discrete_indices] = _log_pdf_discrete(x[:, discrete_indices], discrete_dists)
        weighted_log_pdf = np.sum(log_pdfs, axis=-1) + np.log(self.weights[None, :])
        max_ = weighted_log_pdf.max(axis=1)
        # We need to avoid (-inf) - (-inf) when the probability is zero.
        max_[np.isneginf(max_)] = 0
        with np.errstate(divide="ignore"):  # Suppress warning in log(0).
            return np.log(np.exp(weighted_log_pdf - max_[:, None]).sum(axis=1)) + max_
