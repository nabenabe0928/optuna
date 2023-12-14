from __future__ import annotations

import numpy as np

from optuna.importance.filters._base import BaseFilter
from optuna.trial import FrozenTrial


class QuantileFilter(BaseFilter):
    def __init__(self, quantile: float, is_lower_better: bool):
        if quantile < 0 or quantile > 1:
            raise ValueError(
                f"quantile for {self.__class__.__name__} must be in [0, 1], but got {quantile}."
            )

        self._quantile = quantile
        self._is_lower_better = is_lower_better

    def filter(
        self,
        trials: list[FrozenTrial],
        target_values: np.ndarray,
    ) -> list[FrozenTrial]:
        self._validate_input(trials, target_values)
        if len(target_values.shape) != 1:
            raise ValueError(f"target_values must be 1d array, but got {target_values.shape}.")

        is_lower_better = self._is_lower_better
        quantile = self._quantile if is_lower_better else 1 - self._quantile
        cutoff_value = np.quantile(
            target_values, quantile, method="higher" if is_lower_better else "lower"
        )
        mask = target_values <= cutoff_value if is_lower_better else target_values >= cutoff_value
        return [t for should_be_in, t in zip(mask, trials) if should_be_in]
