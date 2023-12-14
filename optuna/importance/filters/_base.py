from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod

import numpy as np

from optuna.trial import FrozenTrial


class BaseFilter(metaclass=ABCMeta):
    @staticmethod
    def _validate_input(
        trials: list[FrozenTrial],
        target_values: np.ndarray,
    ) -> None:
        if len(trials) == 0 and len(target_values) == 0:
            return

        if len(trials) != len(target_values):
            raise ValueError(
                "The length of trials and target_values must be same, but got "
                f"len(trials)={len(trials)} and len(target_values)={len(target_values)}"
            )

    @abstractmethod
    def filter(
        self,
        trials: list[FrozenTrial],
        target_values: np.ndarray,
    ) -> list[FrozenTrial]:
        raise NotImplementedError
