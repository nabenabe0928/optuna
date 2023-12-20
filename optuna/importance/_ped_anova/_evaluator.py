from __future__ import annotations

from collections.abc import Callable
import warnings

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_filtered_trials
from optuna.importance._base import _get_target_values
from optuna.importance._base import _sort_dict_by_importance
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._ped_anova._scott_parzen_estimator import _build_parzen_estimator
from optuna.importance.filters import QuantileFilter
from optuna.study import Study
from optuna.trial import FrozenTrial


class PedAnovaImportanceEvaluator(BaseImportanceEvaluator):
    """PED-ANOVA importance evaluator.

    Implements the PED-ANOVA hyperparameter importance evaluation algorithm in
    `PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces
      <https://arxiv.org/abs/2304.10255>`_.

    PED-ANOVA fits Parzen estimators of :class:`~optuna.trial.TrialState.COMPLETE` trials better
    than a user-specified baseline. Users can specify the baseline either by a quantile or a value.
    The importance can be interpreted as how important each hyperparameter is to get
    the performance better than baseline.
    Users can also remove trials worse than `cutoff` so that the interpretation removes the bias
    caused by the initial trials.

    For further information about PED-ANOVA algorithm, please refer to the following paper:

    - `PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces
      <https://arxiv.org/abs/2304.10255>`_

    .. note::

        The performance of PED-ANOVA depends on how many trials to consider above baseline.
        To stabilize the analysis, it is preferable to include at least 5 trials above baseline.

    .. note::

        Please refer to the original work available at https://github.com/nabenabe0928/local-anova.

    Args:
        is_lower_better:
            TODO.
        n_steps:
            TODO.
        baseline_quantile:
            TODO.
        baseline_value:
            TODO.
        consider_prior:
            TODO.
        prior_weight:
            TODO.
        categorical_distance_func:
            TODO.
        evaluate_on_local:
            TODO.
    """

    def __init__(
        self,
        is_lower_better: bool,
        *,
        n_steps: int = 50,
        baseline_quantile: float | None = None,
        baseline_value: float | None = None,
        consider_prior: bool = False,
        prior_weight: float = 1.0,
        categorical_distance_func: dict[
            str, Callable[[CategoricalChoiceType, CategoricalChoiceType], float]
        ]
        | None = None,
        evaluate_on_local: bool = True,
        min_n_top_trials: int = 2,
    ):
        if n_steps <= 1:
            raise ValueError(f"`n_steps` must be larger than 1, but got {n_steps}.")
        if baseline_quantile is not None and baseline_value is not None:
            raise ValueError(
                "baseline_quantile and baseline_value cannot be specified simultaneously."
            )
        if min_n_top_trials < 2:
            raise ValueError(
                f"min_n_top_trials must be larger than 1, but got {min_n_top_trials}."
            )
        if baseline_quantile is not None and not (0.0 <= baseline_quantile <= 1.0):
            raise ValueError(f"baseline_quantile must be in [0, 1], but got {baseline_quantile}")
        if baseline_value is None and baseline_quantile is None:
            baseline_quantile = 0.1
            warnings.warn(
                "both baseline_value and baseline_quantile were not specified, "
                f"so baseline_quantile={baseline_quantile} will be used."
            )

        self._n_steps = n_steps
        self._categorical_distance_func = (
            categorical_distance_func if categorical_distance_func is not None else {}
        )
        self._consider_prior = consider_prior
        self._prior_weight = prior_weight
        self._is_lower_better = is_lower_better
        self._min_n_top_trials = min_n_top_trials
        self._baseline_quantile = baseline_quantile
        self._baseline_value = baseline_value
        self._evaluate_on_local = evaluate_on_local

    @staticmethod
    def _validate_and_get_params_and_distributions(
        study: Study,
        params: list[str] | None,
        target: Callable[[FrozenTrial], float] | None,
    ) -> tuple[list[str], dict[str, BaseDistribution]]:
        if target is None and study._is_multi_objective():
            raise ValueError(
                "If the `study` is being used for multi-objective optimization, "
                "please specify the `target`. For example, use "
                "`target=lambda t: t.values[0]` for the first objective value."
            )

        distributions = _get_distributions(study, params=params)
        if params is None:
            params = list(distributions.keys())

        assert params is not None
        return params, distributions

    def _get_trials_better_than_baseline(
        self,
        trials: list[FrozenTrial],
        params: list[str],
        target: Callable[[FrozenTrial], float] | None,
    ) -> list[FrozenTrial]:
        target_values = _get_target_values(trials, target)
        if self._baseline_value is not None:
            mask = (
                target_values <= self._baseline_value
                if self._is_lower_better
                else target_values >= self._baseline_value
            )
            top_trials = [t for should_be_in, t in zip(mask, trials) if should_be_in]
        else:
            assert self._baseline_quantile is not None, "Mypy redefinition."
            quantile_filter = QuantileFilter(
                self._baseline_quantile,
                self._is_lower_better,
                self._min_n_top_trials,
            )
            top_trials = quantile_filter.filter(trials, target_values)

        if len(top_trials) < 5:
            warnings.warn(
                f"The number of trials better than baseline_quantile={self._baseline_quantile} "
                f"and baseline_value={self._baseline_value} is less than 5 and the evaluation"
                " might be inaccurate. Please relax these values."
            )

        return top_trials

    def _compute_pearson_divergence(
        self,
        param_name: str,
        dist: BaseDistribution,
        trials_better_than_baseline: list[FrozenTrial],
        all_trials: list[FrozenTrial],
    ) -> float:
        cat_dist_func = self._categorical_distance_func.get(param_name, None)
        pe_top = _build_parzen_estimator(
            param_name=param_name,
            dist=dist,
            trials=trials_better_than_baseline,
            n_steps=self._n_steps,
            consider_prior=self._consider_prior,
            prior_weight=self._prior_weight,
            categorical_distance_func=cat_dist_func,
        )
        n_grids = pe_top.n_grids
        grids = np.arange(n_grids)
        pdf_top = pe_top.pdf(grids) + 1e-12

        if self._evaluate_on_local:
            # Compute the integral on the local space.
            # It gives us the importances of hyperparameters during the search.
            pe_local = _build_parzen_estimator(
                param_name=param_name,
                dist=dist,
                trials=all_trials,
                n_steps=self._n_steps,
                consider_prior=self._consider_prior,
                prior_weight=self._prior_weight,
                categorical_distance_func=cat_dist_func,
            )
            pdf_local = pe_local.pdf(grids) + 1e-12
        else:
            # Compute the integral on the global space.
            # It gives us the importances of hyperparameters in the search space.
            pdf_local = np.full(n_grids, 1.0 / n_grids)

        return float(pdf_local @ ((pdf_top / pdf_local - 1) ** 2))

    def evaluate(
        self,
        study: Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        params, distributions = self._validate_and_get_params_and_distributions(
            study, params, target
        )

        # PED-ANOVA does not support parameter distributions with a single value,
        # because the importance of such params become zero.
        non_single_distributions = {
            name: dist for name, dist in distributions.items() if not dist.single()
        }
        single_distributions = {
            name: dist for name, dist in distributions.items() if dist.single()
        }
        if len(non_single_distributions) == 0:
            return {}

        trials = _get_filtered_trials(study, params=params, target=target)
        trials_better_than_baseline = self._get_trials_better_than_baseline(trials, params, target)
        importance_sum = 0.0
        param_importances = {}
        for param_name, dist in non_single_distributions.items():
            param_importances[param_name] = self._compute_pearson_divergence(
                param_name,
                dist,
                trials_better_than_baseline=trials_better_than_baseline,
                all_trials=trials,
            )
            importance_sum += param_importances[param_name]

        param_importances = {k: v / importance_sum for k, v in param_importances.items()}
        param_importances.update({k: 0.0 for k in single_distributions})
        return _sort_dict_by_importance(param_importances)
