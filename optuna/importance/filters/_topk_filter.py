from __future__ import annotations

import warnings

import numpy as np

from optuna.importance.filters._base import _get_topk_value
from optuna.importance.filters._base import _validate_filter_trials_input
from optuna.importance.filters._base import BaseFilter
from optuna.trial import FrozenTrial


class TopKFilter(BaseFilter):
    def __init__(self, topk: int, is_lower_better: bool):
        if topk <= 0:
            raise ValueError(f"topk must be a positive integer, but got {topk}.")

        self._topk = topk
        self._is_lower_better = is_lower_better

    def filter(self, trials: list[FrozenTrial], target_values: np.ndarray) -> list[FrozenTrial]:
        _validate_filter_trials_input(trials, target_values)
        target_loss_values = target_values if self._is_lower_better else -target_values
        topk_value = _get_topk_value(
            target_values if self._is_lower_better else -target_values,
            self._topk,
        )
        mask = target_loss_values <= topk_value
        return [t for should_be_in, t in zip(mask, trials) if should_be_in]
