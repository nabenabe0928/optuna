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


def _unique_inverse_2d(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # This function is a quicker version of:
    # np.unique(np.concatenate([a[:, None], b[:, None]], axis=-1), return_inverse=True).
    assert a.shape == b.shape and len(a.shape) == 1
    order_by_b = np.argsort(b)
    # Stable sorting is required for the tie breaking.
    lexsort_order = order_by_b[np.argsort(a[order_by_b], kind="stable")]
    a_order = a[lexsort_order]
    b_order = b[lexsort_order]
    is_first_occurrence = np.empty_like(a_order, dtype=bool)
    is_first_occurrence[0] = True
    is_first_occurrence[1:] = (a_order[1:] != a_order[:-1]) | (b_order[1:] != b_order[:-1])
    inv = np.empty(a_order.size, dtype=int)
    inv[lexsort_order] = np.cumsum(is_first_occurrence) - 1
    return a_order[is_first_occurrence], b_order[is_first_occurrence], inv


def _log_gauss_mass_unique(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This function reduces the log Gaussian probability mass computation by avoiding the
    duplicated evaluations using the np.unique_inverse(...) equivalent operation.
    inv is equivalent to the inverse mapping by np.unique_inverse(np.stack([a, b], axis=-1)).
    """
    if a.size < 100:
        return _truncnorm._log_gauss_mass(a, b)

    a_uniq, b_uniq, inv = _unique_inverse_2d(a_ravel := a.ravel(), b_ravel := b.ravel())
    return _truncnorm._log_gauss_mass(a_uniq, b_uniq)[inv].reshape(a.shape)


def _log_pdf_continuous(x: np.ndarray, dists: list[_BatchedTruncNormDistributions]) -> np.ndarray:
    # x.shape = (batch_size, dim).
    lows = np.asarray([d.low for d in dists])
    highs = np.asarray([d.high for d in dists])
    # {mus, sigmas}.shape = (1, n_kernels, dim).
    mus = np.asarray([d.mu for d in dists]).T[np.newaxis]
    sigmas = np.asarray([d.sigma for d in dists]).T[np.newaxis]
    return _truncnorm.logpdf(
        x=x[:, np.newaxis],
        a=(lows - mus) / sigmas,
        b=(highs - mus) / sigmas,
        loc=mus,
        scale=sigmas,
    ).sum(axis=-1)


def _log_pdf_discrete(
    x: np.ndarray, dists: list[_BatchedDiscreteTruncNormDistributions]
) -> np.ndarray:
    if len(dists) == 0:
        return np.asarray(0.0)

    # x.shape = (batch_size, dim) := (B, D).
    # {mu, sigma}.shape = (1, n_kernels, dim) := (1, N, D).
    n_kernels = len(dists[0].mu)
    res = np.zeros((len(x), n_kernels), dtype=float)
    for i, d in enumerate(dists):
        # NOTE(nabenabe): vectorization --> log_gauss_mass_unique is also possible, but it costs
        # O(NBD log NBD) while the for loop costs O(D * (B log B + N log N)).
        xi_uniq, xi_inv = np.unique(x[:, i], return_inverse=True)
        mu_uniq, sigma_uniq, mu_sigma_inv = _unique_inverse_2d(d.mu, d.sigma)
        res += _log_gauss_mass_unique(
            (xi_uniq[..., np.newaxis] - d.step / 2 - mu_uniq) / sigma_uniq,
            (xi_uniq[..., np.newaxis] + d.step / 2 - mu_uniq) / sigma_uniq,
        )[np.ix_(xi_inv, mu_sigma_inv)]
        res -= _log_gauss_mass_unique(
            (d.low - d.step / 2 - mu_uniq) / sigma_uniq,
            (d.high + d.step / 2 - mu_uniq) / sigma_uniq,
        )[mu_sigma_inv]
    return res


class _MixtureOfProductDistribution(NamedTuple):
    weights: np.ndarray
    distributions: list[_BatchedDistributions]

    def sample(self, rng: np.random.RandomState, batch_size: int) -> np.ndarray:
        active_indices = rng.choice(len(self.weights), p=self.weights, size=batch_size)

        active_mus = np.concatenate([d.mu[active_indices, np.newaxis] for d in self.distributions], axis=1)
        active_sigmas = np.concatenate([d.sigma[active_indices, np.newaxis] for d in self.distributions], axis=1)
        lows = np.array([d.low for d in self.distributions])
        highs = np.array([d.high for d in self.distributions])
        steps = np.array([getattr(d, "step", 0.0) for d in self.distributions])
        ret = _truncnorm.rvs(
            a=(lows - steps / 2 - active_mus) / active_sigmas,
            b=(highs + steps / 2 - active_mus) / active_sigmas,
            loc=active_mus,
            scale=active_sigmas,
            random_state=rng,
        )
        disc_inds = np.nonzero(steps != 0.0)[0]
        ret[:, disc_inds] = np.clip(
            lows[disc_inds] + np.round((ret[:, disc_inds] - lows[disc_inds]) / steps[disc_inds]) * steps[disc_inds],
            lows[disc_inds],
            highs[disc_inds],
        )
        return ret

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
        weighted_log_pdf = np.zeros((len(x), len(self.weights)), dtype=np.float64)
        continuous_dists = []
        continuous_indices = []
        discrete_dists = []
        discrete_indices = []
        for i, d in enumerate(self.distributions):
            if isinstance(d, _BatchedCategoricalDistributions):
                xi = x[:, i]
                weighted_log_pdf += np.log(
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

        weighted_log_pdf += _log_pdf_continuous(x[:, continuous_indices], continuous_dists)
        weighted_log_pdf += _log_pdf_discrete(x[:, discrete_indices], discrete_dists)
        weighted_log_pdf += np.log(self.weights[None, :])
        max_ = weighted_log_pdf.max(axis=1)
        # We need to avoid (-inf) - (-inf) when the probability is zero.
        max_[np.isneginf(max_)] = 0
        with np.errstate(divide="ignore"):  # Suppress warning in log(0).
            return np.log(np.exp(weighted_log_pdf - max_[:, None]).sum(axis=1)) + max_
