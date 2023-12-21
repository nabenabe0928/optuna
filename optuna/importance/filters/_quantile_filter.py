from __future__ import annotations

import warnings

import numpy as np

from optuna.importance.filters._base import _get_cutoff_value_with_warning
from optuna.importance.filters._base import _validate_filter_trials_input
from optuna.importance.filters._base import BaseFilter
from optuna.trial import FrozenTrial


class QuantileFilter(BaseFilter):
    def __init__(self, quantile: float, is_lower_better: bool, min_n_top_trials: int | None = None):
        if quantile < 0 or quantile > 1:
            raise ValueError(
                f"quantile must be in [0, 1], but got {quantile}."
            )
        if min_n_top_trials is not None and min_n_top_trials <= 0:
            raise ValueError(f"min_n_top_trials must be positive, but got {min_n_top_trials}.")

        self._quantile = quantile
        self._is_lower_better = is_lower_better
        self._min_n_top_trials = min_n_top_trials

    def filter(self, trials: list[FrozenTrial], target_values: np.ndarray) -> list[FrozenTrial]:
        _validate_filter_trials_input(trials, target_values)
        target_loss_values = target_values if self._is_lower_better else -target_values
        cutoff_value = np.quantile(target_loss_values, self._quantile, method="higher")
        cutoff_value = self._get_cutoff_value_with_warning(
            target_loss_values,
            cutoff_value=cutoff_value,
            cond_name="quantile",
        )
        mask = target_loss_values <= cutoff_value
        return [t for should_be_in, t in zip(mask, trials) if should_be_in]
