from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
import warnings

import numpy as np

from optuna.trial import FrozenTrial


def _validate_filter_trials_input(trials: list[FrozenTrial], target_values: np.ndarray) -> None:
    if len(trials) == 0 and len(target_values) == 0:
        return

    if len(trials) != len(target_values):
        raise ValueError(
            "The length of trials and target_values must be same, but got "
            f"len(trials)={len(trials)} and len(target_values)={len(target_values)}"
        )

    if len(target_values.shape) != 1:
        raise ValueError(f"target_values must be 1d array, but got {target_values.shape}.")


def _get_topk_value(target_loss_values: np.ndarray, topk: int) -> float:
    if topk > target_loss_values.size or topk < 1:
        raise ValueError(f"topk must be in [1, {target_loss_values.size}], but got {topk}.")
    topk_index = topk - 1
    return np.partition(target_loss_values, topk)[topk - 1]


class BaseFilter(metaclass=ABCMeta):
    _min_n_top_trials: int | None

    def _get_cutoff_value_with_warning(
        self,
        target_loss_values: np.ndarray,
        cutoff_value: float,
        cond_name: str,
    ) -> float:
        if self._min_n_top_trials is None:
            return cutoff_value

        top_value = _get_topk_value(target_loss_values, self._min_n_top_trials)
        if cutoff_value < top_value:
            value_instead = top_value if is_lower_better else -top_value
            cond_value = getattr(self, f"_{cond_name}")
            msg = [
                f"The given {cond_name}={cond_val} was too tight to have ",
                f"{self._min_n_top_trials} trials after applying {self.__class__.__name__}, ",
                f"so {value_instead} was used as cutoff_value.",
            ]
            warnings.warn("".join(msg))

        return max(cutoff_value, top_value)

    @abstractmethod
    def filter(self, trials: list[FrozenTrial], target_values: np.ndarray) -> list[FrozenTrial]:
        raise NotImplementedError
