from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optuna.importance.filters._base import _get_topk_value
from optuna.importance.filters._base import BaseFilter
from optuna.importance.filters._base import FilterRunner
from optuna.trial import FrozenTrial


class TopKFilter(BaseFilter):
    def __init__(
        self,
        topk: int,
        is_lower_better: bool,
        target: Callable[[FrozenTrial], float] | None = None,
    ):
        if topk <= 0:
            raise ValueError(f"topk must be a positive integer, but got {topk}.")

        self._filter_runner = FilterRunner(
            is_lower_better=is_lower_better,
            cond_name="topk",
            cond_value=topk,
            min_n_top_trials=None,
            target=target,
            filter_name=self.__class__.__name__,
            cutoff_value_calculate_method=self._calculate_cutoff_value,
        )
        self._topk = topk

    def _calculate_cutoff_value(self, target_loss_values: np.ndarray) -> float:
        return _get_topk_value(target_loss_values, self._topk)
