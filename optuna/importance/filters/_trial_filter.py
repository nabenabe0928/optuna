from __future__ import annotations

from collections.abc import Callable

from optuna.importance.filters import CustomFilter
from optuna.importance.filters import CutOffFilter
from optuna.importance.filters import QuantileFilter
from optuna.importance.filters import TopKFilter
from optuna.trial import FrozenTrial


def get_trial_filter(
    topk: int | None = None,
    quantile: float | None = None,
    cutoff_value: float | None = None,
    custom_filter_trial: Callable[[FrozenTrial], bool] | None = None,
    is_lower_better: bool | None = None,
    min_n_top_trials: int | None = None,
    target: Callable[[FrozenTrial], float] | None = None,
) -> Callable[[list[FrozenTrial]], list[FrozenTrial]]:
    """Get trial filter.

    Args:
        topk:
            Filter top `k` trials.
        quantile:
            Filter top `quantile * 100`% trials.
            For example, `quantile=0.1` means trials better than top-10% will be filtered.
        cutoff_value:
            Filter trials with `target_value` better than `cutoff_value`.
        custom_filter_trial:
            A function that returns True or False given `trial`.
            If True, `trial` will remain in the return.
            For example, if you have a list of trial numbers to filter `trial_numbers`,
            we can do `lambda trial: trial.number in trial_numbers_set`
            given `trial_numbers_set = set(trial_numbers)`.
        is_lower_better:
            Whether `target_value` is better when it is lower.
        min_n_top_trials:
            The minimum number of trials to be included in the filtered trials.
        target:
            A function to specify the value to evaluate importances.
            If it is :obj:`None` and ``study`` is being used for single-objective optimization,
            the objective values are used. Can also be used for other trial attributes, such as
            the duration, like ``target=lambda t: t.duration.total_seconds()``.

    Returns:
        A list of filtered trials.
    """

    arg_names = ", ".join(["topk", "quantile", "cutoff_value", "custom_filter_trial"])
    if sum(v is not None for v in [topk, quantile, cutoff_value, custom_filter_trial]) > 1:
        raise ValueError(f"Only one of {{{arg_names}}} can be specified.")
    if all(v is None for v in [topk, quantile, cutoff_value, custom_filter_trial]):
        raise ValueError(f"One of {{{arg_names}}} must be specified.")

    if any(v is not None for v in [topk, quantile, cutoff_value]) and is_lower_better is None:
        raise ValueError("is_lower_better must be specified, but got None.")

    if topk is not None:
        assert is_lower_better is not None, "MyPy Redefinition."
        return TopKFilter(topk, is_lower_better, target).filter
    elif quantile is not None:
        assert is_lower_better is not None, "MyPy Redefinition."
        return QuantileFilter(quantile, is_lower_better, min_n_top_trials, target).filter
    elif cutoff_value is not None:
        assert is_lower_better is not None, "MyPy Redefinition."
        return CutOffFilter(cutoff_value, is_lower_better, min_n_top_trials, target).filter
    elif custom_filter_trial is not None:
        return CustomFilter(custom_filter_trial).filter
    else:
        assert False, "Should not be reached."
