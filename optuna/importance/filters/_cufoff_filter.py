from __future__ import annotations

import warnings

import numpy as np

from optuna.importance.filters._base import _validate_filter_trials_input
from optuna.importance.filters._base import BaseFilter
from optuna.trial import FrozenTrial


class CutOffFilter(BaseFilter):
    def __init__(
        self,
        cutoff_value: float,
        is_lower_better: bool,
        min_n_top_trials: int | None = None,
    ):
        self._cutoff_value = cutoff_value
        self._is_lower_better = is_lower_better
        self._min_n_top_trials = min_n_top_trials

    def filter(self, trials: list[FrozenTrial], target_values: np.ndarray) -> list[FrozenTrial]:
        _validate_filter_trials_input(trials, target_values)
        target_loss_values = target_values if self._is_lower_better else -target_values
        cutoff_value = self._cutoff_value
        cutoff_value = self._get_cutoff_value_with_warning(
            target_loss_values,
            cutoff_value=cutoff_value,
            cond_name="cutoff_value",
        )
        mask = target_loss_values <= cutoff_value
        return [t for should_be_in, t in zip(mask, trials) if should_be_in]
