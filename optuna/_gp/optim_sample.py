from __future__ import annotations

import numpy as np

from optuna._gp.acqf import BaseAcquisitionFunc
from optuna._gp.search_space import sample_normalized_params


def optimize_acqf_sample(
    acqf: BaseAcquisitionFunc, *, n_samples: int = 2048, rng: np.random.RandomState | None = None
) -> tuple[np.ndarray, float]:
    # Normalized parameter values are sampled.
    xs = sample_normalized_params(n_samples, acqf_params.search_space, rng=rng)
    res, _ = acqf(xs, with_grad=False)

    best_i = np.argmax(res)
    return xs[best_i, :], res[best_i]
