from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optuna.importance.filters._base import BaseFilter
from optuna.trial import FrozenTrial


class CustomFilter(BaseFilter):
    def __init__(self, filter_trial: Callable[[FrozenTrial], bool]) -> None:
        self._filter_runner = None
        self._filter_trial = filter_trial

    def filter(self, trials: list[FrozenTrial]) -> list[FrozenTrial]:
        return list(filter(self._filter_trial, trials))

    def _calculate_cutoff_value(self, target_loss_values: np.ndarray) -> float:
        return np.nan
