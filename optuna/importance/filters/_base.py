from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from collections.abc import Callable
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

    return np.partition(target_loss_values, topk - 1)[topk - 1]


class FilterRunner:
    def __init__(
        self,
        is_lower_better: bool,
        cond_name: str,
        cond_value: float,
        min_n_top_trials: int | None,
        filter_name: str,
        cutoff_value_calculate_method: Callable[[np.ndarray], float],
    ):
        self._is_lower_better = is_lower_better
        self._cond_name = cond_name
        self._cond_value = cond_value
        self._min_n_top_trials = min_n_top_trials
        self._filter_name = filter_name
        self._cutoff_value_calculate_method = cutoff_value_calculate_method

    def _get_cutoff_value_with_warning(self, target_loss_values: np.ndarray) -> float:
        cutoff_value = self._cutoff_value_calculate_method(target_loss_values)
        if self._min_n_top_trials is None:
            return cutoff_value

        top_value = _get_topk_value(target_loss_values, self._min_n_top_trials)
        if cutoff_value < top_value:
            value_instead = top_value if self._is_lower_better else -top_value
            msg = [
                f"The given {self._cond_name}={self._cond_value} was too tight to have ",
                f"{self._min_n_top_trials} trials after applying {self._filter_name}, ",
                f"so {value_instead} was used as cutoff_value.",
            ]
            warnings.warn("".join(msg))

        return max(cutoff_value, top_value)

    def filter(self, trials: list[FrozenTrial], target_values: np.ndarray) -> list[FrozenTrial]:
        _validate_filter_trials_input(trials, target_values)
        target_loss_values = target_values if self._is_lower_better else -target_values
        mask = target_loss_values <= self._get_cutoff_value_with_warning(target_loss_values)
        return [t for should_be_in, t in zip(mask, trials) if should_be_in]


class BaseFilter(metaclass=ABCMeta):
    _filter_runner: FilterRunner

    @abstractmethod
    def _calculate_cutoff_value(self, target_loss_values: np.ndarray) -> float:
        raise NotImplementedError

    def filter(self, trials: list[FrozenTrial], target_values: np.ndarray) -> list[FrozenTrial]:
        """Filter trials based on target_values.

        Args:
            trials:
                A list of trials to which the filter is applied.
            target_values:
                An array of target_values that defines the value of each trial.
                The shape must be (len(trials), ) and target_values[i] is the value of trials[i].

        Returns:
            A list of filtered trials.
        """

        return self._filter_runner.filter(trials, target_values)
