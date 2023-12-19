from __future__ import annotations

import warnings

import numpy as np

from optuna.importance.filters._base import BaseFilter
from optuna.trial import FrozenTrial


class QuantileFilter(BaseFilter):
    def __init__(
        self,
        quantile: float,
        is_lower_better: bool,
        min_n_top_trials: int | None = None,
    ):
        if quantile < 0 or quantile > 1:
            raise ValueError(
                f"quantile for {self.__class__.__name__} must be in [0, 1], but got {quantile}."
            )
        if min_n_top_trials is not None and min_n_top_trials <= 0:
            raise ValueError(f"min_n_top_trials must be positive, but got {min_n_top_trials}.")

        self._quantile = quantile
        self._is_lower_better = is_lower_better
        self._min_n_top_trials = min_n_top_trials

    def filter(
        self,
        trials: list[FrozenTrial],
        target_values: np.ndarray,
    ) -> list[FrozenTrial]:
        self._validate_input(trials, target_values)
        if len(target_values.shape) != 1:
            raise ValueError(f"target_values must be 1d array, but got {target_values.shape}.")

        _target_values = target_values if self._is_lower_better else -target_values
        cutoff_value = np.quantile(_target_values, self._quantile, method="higher")
        if self._min_n_top_trials is not None:
            top_index = self._min_n_top_trials - 1
            top_value = np.partition(_target_values, top_index)[top_index]
            if cutoff_value < top_value:
                msg = [
                    f"The given quantile {self._quantile} was too tight to have ",
                    f"{self._min_n_top_trials} after applying {self.__class__.__name__}, ",
                    f"so {top_value if self._is_lower_better else -top_value} was used for ",
                    "the threshold.",
                ]
                warnings.warn("".join(msg))
                cutoff_value = top_value

        mask = target_values <= cutoff_value
        return [t for should_be_in, t in zip(mask, trials) if should_be_in]
