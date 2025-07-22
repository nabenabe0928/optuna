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


class _MixtureOfProductDistribution(NamedTuple):
    weights: np.ndarray
    distributions: list[_BatchedDistributions]

    def sample(self, rng: np.random.RandomState, batch_size: int) -> np.ndarray:
        active_indices = rng.choice(len(self.weights), p=self.weights, size=batch_size)

        ret = np.empty((batch_size, len(self.distributions)), dtype=np.float64)
        for i, d in enumerate(self.distributions):
            if isinstance(d, _BatchedCategoricalDistributions):
                active_weights = d.weights[active_indices, :]
                rnd_quantile = rng.rand(batch_size)
                cum_probs = np.cumsum(active_weights, axis=-1)
                assert np.isclose(cum_probs[:, -1], 1).all()
                cum_probs[:, -1] = 1  # Avoid numerical errors.
                ret[:, i] = np.sum(cum_probs < rnd_quantile[:, None], axis=-1)
            elif isinstance(d, _BatchedTruncNormDistributions):
                active_mus = d.mu[active_indices]
                active_sigmas = d.sigma[active_indices]
                ret[:, i] = _truncnorm.rvs(
                    a=(d.low - active_mus) / active_sigmas,
                    b=(d.high - active_mus) / active_sigmas,
                    loc=active_mus,
                    scale=active_sigmas,
                    random_state=rng,
                )
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                active_mus = d.mu[active_indices]
                active_sigmas = d.sigma[active_indices]
                samples = _truncnorm.rvs(
                    a=(d.low - d.step / 2 - active_mus) / active_sigmas,
                    b=(d.high + d.step / 2 - active_mus) / active_sigmas,
                    loc=active_mus,
                    scale=active_sigmas,
                    random_state=rng,
                )
                ret[:, i] = np.clip(
                    d.low + np.round((samples - d.low) / d.step) * d.step, d.low, d.high
                )
            else:
                assert False

        return ret

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        batch_size, n_vars = x.shape
        log_pdfs = np.empty((batch_size, len(self.weights), n_vars), dtype=np.float64)
        cont_dists = []
        cont_inds = []
        for i, d in enumerate(self.distributions):
            if isinstance(d, _BatchedCategoricalDistributions):
                xi = x[:, i, np.newaxis, np.newaxis].astype(np.int64)
                log_pdfs[:, :, i] = np.log(np.take_along_axis(d.weights[np.newaxis], xi, axis=-1))[
                    ..., 0
                ]
            elif isinstance(d, _BatchedTruncNormDistributions):
                cont_dists.append(d)
                cont_inds.append(i)
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                xi = x[:, i]
                lower_limit = d.low - d.step / 2
                upper_limit = d.high + d.step / 2
                x_lower = np.maximum(xi - d.step / 2, lower_limit)
                x_upper = np.minimum(xi + d.step / 2, upper_limit)
                log_gauss_mass = _truncnorm._log_gauss_mass(
                    (x_lower[:, None] - d.mu[None, :]) / d.sigma[None, :],
                    (x_upper[:, None] - d.mu[None, :]) / d.sigma[None, :],
                )
                log_p_accept = _truncnorm._log_gauss_mass(
                    (d.low - d.step / 2 - d.mu[None, :]) / d.sigma[None, :],
                    (d.high + d.step / 2 - d.mu[None, :]) / d.sigma[None, :],
                )
                log_pdfs[:, :, i] = log_gauss_mass - log_p_accept
            else:
                assert False

        mus_cont = np.asarray([d.mu for d in cont_dists]).T
        sigmas_cont = np.asarray([d.sigma for d in cont_dists]).T
        log_pdfs[..., cont_inds] = _truncnorm.logpdf(
            x[:, np.newaxis, cont_inds],
            (np.asarray([d.low for d in cont_dists]) - mus_cont) / sigmas_cont,
            (np.asarray([d.high for d in cont_dists]) - mus_cont) / sigmas_cont,
            loc=mus_cont,
            scale=sigmas_cont,
        )
        weighted_log_pdf = np.sum(log_pdfs, axis=-1) + np.log(self.weights[None, :])
        max_ = weighted_log_pdf.max(axis=1)
        # We need to avoid (-inf) - (-inf) when the probability is zero.
        max_[np.isneginf(max_)] = 0
        with np.errstate(divide="ignore"):  # Suppress warning in log(0).
            return np.log(np.exp(weighted_log_pdf - max_[:, None]).sum(axis=1)) + max_
