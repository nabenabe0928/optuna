from __future__ import annotations

import numpy as np

from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator


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
