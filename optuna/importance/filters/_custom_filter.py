from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optuna.importance.filters._base import BaseFilter
from optuna.importance.filters._base import FilterRunner
from optuna.trial import FrozenTrial


class CustomFilter(BaseFilter):
    def __init__(self) -> None:
        self._filter_runner = FilterRunner(
            is_lower_better=False,  # Filter True.
            cond_name="None",
            cond_value=np.nan,
            min_n_top_trials=None,
            filter_name=self.__class__.__name__,
            cutoff_value_calculate_method=self._calculate_cutoff_value,
        )

    def filter(self, trials: list[FrozenTrial], target_values: np.ndarray) -> list[FrozenTrial]:
        if target_values.dtype != "bool":
            raise ValueError(f"target_values must be bool array for {self.__class__.__name__}.")

        mask = target_values.astype(float)
        return super().filter(trials, mask)

    def filter_by_mask(self, trials: list[FrozenTrial], mask: np.ndarray) -> list[FrozenTrial]:
        return self.filter(trials, mask)

    def filter_by_method(
        self,
        trials: list[FrozenTrial],
        filter_trial: Callable[[FrozenTrial], bool],
    ) -> list[FrozenTrial]:
        return list(filter(filter_trial, trials))

    def _calculate_cutoff_value(self, target_loss_values: np.ndarray) -> float:
        return 0.5
