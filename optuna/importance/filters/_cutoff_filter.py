from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optuna.importance.filters._base import BaseFilter
from optuna.importance.filters._base import FilterRunner
from optuna.trial import FrozenTrial


class CutOffFilter(BaseFilter):
    def __init__(
        self,
        cutoff_value: float,
        is_lower_better: bool,
        min_n_top_trials: int | None = None,
        target: Callable[[FrozenTrial], float] | None = None,
    ):
        self._filter_runner = FilterRunner(
            is_lower_better=is_lower_better,
            cond_name="cutoff_value",
            cond_value=cutoff_value,
            min_n_top_trials=min_n_top_trials,
            target=target,
            filter_name=self.__class__.__name__,
            cutoff_value_calculate_method=self._calculate_cutoff_value,
        )
        self._cutoff_value = cutoff_value if is_lower_better else -cutoff_value

    def _calculate_cutoff_value(self, target_loss_values: np.ndarray) -> float:
        return self._cutoff_value