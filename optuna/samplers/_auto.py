from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import TYPE_CHECKING

from optuna.samplers import BaseSampler
from optuna.samplers import TPESampler
from optuna.search_space import IntersectionSearchSpace


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial
    from optuna.trial import TrialState


class AutoSampler(BaseSampler):
    """Sampler automatically choosing an appropriate sampler based on search space.

    This sampler is convenient when you are unsure what sampler to use.

    Example:

        .. testcode::

            import optuna
            from optuna.samplers import AutoSampler


            def objective(trial):
                x = trial.suggest_float("x", -5, 5)
                return x**2


            study = optuna.create_study(sampler=AutoSampler())
            study.optimize(objective, n_trials=10)

    .. note::
        This sampler might require ``scipy``, ``torch``, and ``cmaes``.
        You can install these dependencies with ``pip install scipy torch cmaes``.

    Args:
        seed: Seed for random number generator.

    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._init_search_space: dict[str, BaseDistribution] | None = None
        self._sampler: BaseSampler = TPESampler(seed=seed)

    def reseed_rng(self) -> None:
        self._sampler.reseed_rng()

    def _determine_sampler(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> None:
        if self._init_search_space != search_space:
            # NOTE(nabenabe): The statement above is always true for Trial#1.
            # Under discussion. (set(self._init_search_space) == set(search_space))
            if not isinstance(self._sampler, TPESampler):
                self._sampler = TPESampler(seed=self._seed)

            return

        if ...:
            pass
        if trial.number > 100 and not isinstance(...):
            pass
        elif trial.number > 1000 and not isinstance(...):
            pass

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space = IntersectionSearchSpace().calculate(study)
        if len(search_space) == 0:
            return {}

        if self._init_search_space is None:
            self._init_search_space = search_space

        self._determine_sampler(study, trial, search_space)
        return self._sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return self._sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._sampler.sample_independent(study, trial, param_name, param_distribution)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._sampler.after_trial(study, trial, state, values)
