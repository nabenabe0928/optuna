from __future__ import annotations

import math
from typing import Any
from typing import cast
from typing import TYPE_CHECKING

import numpy as np

from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._random import RandomSampler
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.search_space import IntersectionSearchSpace
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.study import Study


EPS = 1e-12


def default_weights(x: int) -> np.ndarray:
    if x == 0:
        return np.asarray([])
    elif x < 25:
        return np.ones(x)
    else:
        ramp = np.linspace(1.0 / x, 1.0, num=x - 25)
        flat = np.ones(25)
        return np.concatenate([ramp, flat], axis=0)


class TPESampler(BaseSampler):
    def __init__(
        self,
        *,
        seed: int | None = None,
        multivariate: bool = False,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
    ) -> None:
        self._n_startup_trials = 10
        self._n_ei_candidates = 24
        self._rng = LazyRandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)
        self._multivariate = multivariate
        self._search_space = IntersectionSearchSpace()
        self._constraints_func = constraints_func

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._random_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        if not self._multivariate:
            return {}
        complete_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        return complete_trials[0].distributions if complete_trials else {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}
        trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True)
        if len(trials) < self._n_startup_trials:
            return {}  # Fall back to sample_independent.
        return self._sample(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True)
        if len(trials) < self._n_startup_trials:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )
        return self._sample(study, trial, {param_name: param_distribution})[param_name]

    def _get_internal_repr(
        self, trials: list[FrozenTrial], search_space: dict[str, BaseDistribution]
    ) -> dict[str, np.ndarray]:
        values: dict[str, list[float]] = {param_name: [] for param_name in search_space}
        for trial in trials:
            params = trial.params
            for param_name, distribution in search_space.items():
                param = params[param_name]
                values[param_name].append(distribution.to_internal_repr(param))
        return {k: np.asarray(v) for k, v in values.items()}

    def _sample(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        trials = study._get_trials(deepcopy=False, states=[TrialState.COMPLETE], use_cache=True)
        # We divide data into below and above.
        n_below = min(math.ceil(0.1 * len(trials)), 25)
        below_trials, above_trials = _split_trials(
            study, trials, n_below, self._constraints_func is not None
        )
        mpe_below = _ParzenEstimator(
            self._get_internal_repr(below_trials, search_space), search_space, default_weights
        )
        mpe_above = _ParzenEstimator(
            self._get_internal_repr(above_trials, search_space), search_space, default_weights
        )
        samples_below = mpe_below.sample(self._rng.rng, self._n_ei_candidates)
        log_likelihoods_below = mpe_below.log_pdf(samples_below)
        log_likelihoods_above = mpe_above.log_pdf(samples_below)
        best_idx = np.argmax(log_likelihoods_below - log_likelihoods_above)
        return {
            k: search_space[k].to_external_repr(v[best_idx].item())
            for k, v in samples_below.items()
        }

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._random_sampler.before_trial(study, trial)

    def after_trial(
        self, study: Study, trial: FrozenTrial, state: TrialState, values: Sequence[float] | None
    ) -> None:
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
        self._random_sampler.after_trial(study, trial, state, values)


def _split_trials(
    study: Study, trials: list[FrozenTrial], n_below: int, constraints_enabled: bool
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    complete_trials = []
    infeasible_trials = []
    for trial in trials:
        if constraints_enabled and _get_infeasible_trial_score(trial) > 0:
            infeasible_trials.append(trial)
        elif trial.state == TrialState.COMPLETE:
            complete_trials.append(trial)
        else:
            assert False

    n_below = min(n_below, len(complete_trials))
    reverse = study.direction != StudyDirection.MINIMIZE
    sorted_complete = sorted(
        complete_trials, key=lambda trial: cast(float, trial.value), reverse=reverse
    )
    below_complete, above_complete = sorted_complete[:n_below], sorted_complete[n_below:]

    n_below = min(max(0, n_below - len(below_complete)), len(infeasible_trials))
    sorted_infeasible = sorted(infeasible_trials, key=_get_infeasible_trial_score)
    below_infeasible, above_infeasible = sorted_infeasible[:n_below], sorted_infeasible[n_below:]
    return (
        sorted(below_complete + below_infeasible, key=lambda trial: trial.number),
        sorted(above_complete + above_infeasible, key=lambda trial: trial.number),
    )


def _get_infeasible_trial_score(trial: FrozenTrial) -> float:
    constraint = trial.system_attrs.get(_CONSTRAINTS_KEY)
    assert constraint is not None
    # Violation values of infeasible dimensions are summed up.
    return sum(v for v in constraint if v > 0)
