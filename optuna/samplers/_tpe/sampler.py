from __future__ import annotations

import math
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Sequence
import warnings

import numpy as np

from optuna._hypervolume import WFG
from optuna._hypervolume.hssp import _solve_hssp
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.exceptions import ExperimentalWarning
from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._random import RandomSampler
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorList
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.search_space import IntersectionSearchSpace
from optuna.search_space.group_decomposed import _GroupDecomposedSearchSpace
from optuna.search_space.group_decomposed import _SearchSpaceGroup
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


EPS = 1e-12
_logger = get_logger(__name__)


def _infer_n_constraints(trials: list[FrozenTrial]) -> int:
    n_constraints = max(
        len(t.system_attrs[_CONSTRAINTS_KEY])
        for t in trials if _CONSTRAINTS_KEY in t.system_attrs
    )
    for t in trials:
        if _CONSTRAINTS_KEY not in t.system_attrs:
            continue

        n_constraints_of_trial = len(t.system_attrs[_CONSTRAINTS_KEY])
        if n_constraints_of_trial < n_constraints:
            raise ValueError(
                "The number of constraints must be consistent during an optimization, but got "
                f"n_constraints={n_constraints_of_trial} at Trial#{t.number}."
            )

    return n_constraints


def default_gamma(x: int) -> int:
    return min(int(np.ceil(0.1 * x)), 25)


def hyperopt_default_gamma(x: int) -> int:
    return min(int(np.ceil(0.25 * np.sqrt(x))), 25)


def ctpe_default_gamma(
    x: int, study: Study, trials: list[FrozenTrial], gamma: Callable[[int], int]
) -> int:
    n_constraints = _infer_n_constraints(trials)
    if n_constraints == 0:
        return gamma(x)

    sorted_trials, _ = _split_trials(
        study=study,
        trials=trials,
        n_below=len(trials),
        constraints_enabled=False,
        order_by="value",
    )
    feasible_masks = [
        # If `_CONSTRAINTS_KEY` does not exist in a `trial`, consider it as infeasible.
        all(c <= 0 for c in t.system_attrs.get(_CONSTRAINTS_KEY, [1.0]))
        for t in sorted_trials
    ]
    n_below = gamma(x)
    # sorted_trials[gamma_modified] is the top gamma-th feasible trial.
    n_below_modified = int(np.searchsorted(np.cumsum(feasible_masks), n_below, side="left")) + 1
    return min(n_below_modified, len(trials))


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
    """Sampler using TPE (Tree-structured Parzen Estimator) algorithm.

    This sampler is based on *independent sampling*.
    See also :class:`~optuna.samplers.BaseSampler` for more details of 'independent sampling'.

    On each trial, for each parameter, TPE fits one Gaussian Mixture Model (GMM) ``l(x)`` to
    the set of parameter values associated with the best objective values, and another GMM
    ``g(x)`` to the remaining parameter values. It chooses the parameter value ``x`` that
    maximizes the ratio ``l(x)/g(x)``.

    For further information about TPE algorithm, please refer to the following papers:

    - `Algorithms for Hyper-Parameter Optimization
      <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`_
    - `Making a Science of Model Search: Hyperparameter Optimization in Hundreds of
      Dimensions for Vision Architectures <http://proceedings.mlr.press/v28/bergstra13.pdf>`_
    - `Tree-Structured Parzen Estimator: Understanding Its Algorithm Components and Their Roles for
      Better Empirical Performance <https://arxiv.org/abs/2304.11127>`_

    For multi-objective TPE (MOTPE), please refer to the following papers:

    - `Multiobjective Tree-Structured Parzen Estimator for Computationally Expensive Optimization
      Problems <https://dl.acm.org/doi/10.1145/3377930.3389817>`_
    - `Multiobjective Tree-Structured Parzen Estimator <https://doi.org/10.1613/jair.1.13188>`_

    Example:
        An example of a single-objective optimization is as follows:

        .. testcode::

            import optuna
            from optuna.samplers import TPESampler


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return x**2


            study = optuna.create_study(sampler=TPESampler())
            study.optimize(objective, n_trials=10)

    .. note::
        :class:`~optuna.samplers.TPESampler` can handle a multi-objective task as well and
        the following shows an example:

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                f1 = x**2 + y
                f2 = -((x - 2) ** 2 + y)
                return f1, f2


            # We minimize the first objective and maximize the second objective.
            sampler = optuna.samplers.TPESampler()
            study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)
            study.optimize(objective, n_trials=100)

    Args:
        consider_prior:
            Enhance the stability of Parzen estimator by imposing a Gaussian prior when
            :obj:`True`. The prior is only effective if the sampling distribution is
            either :class:`~optuna.distributions.FloatDistribution`,
            or :class:`~optuna.distributions.IntDistribution`.
        prior_weight:
            The weight of the prior. This argument is used in
            :class:`~optuna.distributions.FloatDistribution`,
            :class:`~optuna.distributions.IntDistribution`, and
            :class:`~optuna.distributions.CategoricalDistribution`.
        consider_magic_clip:
            Enable a heuristic to limit the smallest variances of Gaussians used in
            the Parzen estimator.
        consider_endpoints:
            Take endpoints of domains into account when calculating variances of Gaussians
            in Parzen estimator. See the original paper for details on the heuristics
            to calculate the variances.
        n_startup_trials:
            The random sampling is used instead of the TPE algorithm until the given number
            of trials finish in the same study.
        n_ei_candidates:
            Number of candidate samples used to calculate the expected improvement.
        gamma:
            A function that takes the number of finished trials and returns the number
            of trials to form a density function for samples with low grains.
            See the original paper for more details.
        weights:
            A function that takes the number of finished trials and returns a weight for them.
            See `Making a Science of Model Search: Hyperparameter Optimization in Hundreds of
            Dimensions for Vision Architectures <http://proceedings.mlr.press/v28/bergstra13.pdf>`_
            for more details.

            .. note::
                In the multi-objective case, this argument is only used to compute the weights of
                bad trials, i.e., trials to construct `g(x)` in the `paper
                <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`_
                ). The weights of good trials, i.e., trials to construct `l(x)`, are computed by a
                rule based on the hypervolume contribution proposed in the `paper of MOTPE
                <https://dl.acm.org/doi/10.1145/3377930.3389817>`_.
        seed:
            Seed for random number generator.
        multivariate:
            If this is :obj:`True`, the multivariate TPE is used when suggesting parameters.
            The multivariate TPE is reported to outperform the independent TPE. See `BOHB: Robust
            and Efficient Hyperparameter Optimization at Scale
            <http://proceedings.mlr.press/v80/falkner18a.html>`_ for more details.

            .. note::
                Added in v2.2.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.2.0.
        group:
            If this and ``multivariate`` are :obj:`True`, the multivariate TPE with the group
            decomposed search space is used when suggesting parameters.
            The sampling algorithm decomposes the search space based on past trials and samples
            from the joint distribution in each decomposed subspace.
            The decomposed subspaces are a partition of the whole search space. Each subspace
            is a maximal subset of the whole search space, which satisfies the following:
            for a trial in completed trials, the intersection of the subspace and the search space
            of the trial becomes subspace itself or an empty set.
            Sampling from the joint distribution on the subspace is realized by multivariate TPE.
            If ``group`` is :obj:`True`, ``multivariate`` must be :obj:`True` as well.

            .. note::
                Added in v2.8.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.8.0.

            Example:

            .. testcode::

                import optuna


                def objective(trial):
                    x = trial.suggest_categorical("x", ["A", "B"])
                    if x == "A":
                        return trial.suggest_float("y", -10, 10)
                    else:
                        return trial.suggest_int("z", -10, 10)


                sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
                study = optuna.create_study(sampler=sampler)
                study.optimize(objective, n_trials=10)
        warn_independent_sampling:
            If this is :obj:`True` and ``multivariate=True``, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler.
            If ``multivariate=False``, this flag has no effect.
        constant_liar:
            If :obj:`True`, penalize running trials to avoid suggesting parameter configurations
            nearby.

            .. note::
                Abnormally terminated trials often leave behind a record with a state of
                ``RUNNING`` in the storage.
                Such "zombie" trial parameters will be avoided by the constant liar algorithm
                during subsequent sampling.
                When using an :class:`~optuna.storages.RDBStorage`, it is possible to enable the
                ``heartbeat_interval`` to change the records for abnormally terminated trials to
                ``FAIL``.

            .. note::
                It is recommended to set this value to :obj:`True` during distributed
                optimization to avoid having multiple workers evaluating similar parameter
                configurations. In particular, if each objective function evaluation is costly
                and the durations of the running states are significant, and/or the number of
                workers is high.

            .. note::
                Added in v2.8.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.8.0.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraints is violated. A value equal to or smaller than 0 is considered feasible.
            If ``constraints_func`` returns more than one value for a trial, that trial is
            considered feasible if and only if all values are equal to 0 or smaller.

            The ``constraints_func`` will be evaluated after each successful trial.
            The function won't be called when trials fail or they are pruned, but this behavior is
            subject to change in the future releases.

            .. note::
                Added in v3.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice.
                See https://github.com/optuna/optuna/releases/tag/v3.0.0.
        categorical_distance_func:
            A dictionary of distance functions for categorical parameters. The key is the name of
            the categorical parameter and the value is a distance function that takes two
            :class:`~optuna.distributions.CategoricalChoiceType` s and returns a :obj:`float`
            value. The distance function must return a non-negative value.

            While categorical choices are handled equally by default, this option allows users to
            specify prior knowledge on the structure of categorical parameters. When specified,
            categorical choices closer to current best choices are more likely to be sampled.

            .. note::
                Added in v3.4.0 as an experimental feature. The interface may change in newer
                versions without prior notice.
                See https://github.com/optuna/optuna/releases/tag/v3.4.0.
    """

    def __init__(
        self,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
        consider_magic_clip: bool = True,
        consider_endpoints: bool = False,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        gamma: Callable[[int], int] = default_gamma,
        weights: Callable[[int], np.ndarray] = default_weights,
        seed: Optional[int] = None,
        *,
        multivariate: bool = False,
        group: bool = False,
        warn_independent_sampling: bool = True,
        constant_liar: bool = False,
        constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
        ctpe: bool = False,
        categorical_distance_func: Optional[
            dict[str, Callable[[CategoricalChoiceType, CategoricalChoiceType], float]]
        ] = None,
    ) -> None:
        self._parzen_estimator_parameters = _ParzenEstimatorParameters(
            consider_prior,
            prior_weight,
            consider_magic_clip,
            consider_endpoints,
            weights,
            multivariate,
            categorical_distance_func or {},
        )
        self._n_startup_trials = n_startup_trials
        self._n_ei_candidates = n_ei_candidates
        self._gamma = gamma

        self._warn_independent_sampling = warn_independent_sampling
        self._rng = LazyRandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)

        self._multivariate = multivariate
        self._group = group
        self._group_decomposed_search_space: Optional[_GroupDecomposedSearchSpace] = None
        self._search_space_group: Optional[_SearchSpaceGroup] = None
        self._search_space = IntersectionSearchSpace(include_pruned=True)
        self._constant_liar = constant_liar
        self._constraints_func = constraints_func
        self._ctpe = ctpe and constraints_func is not None

        if multivariate:
            warnings.warn(
                "``multivariate`` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

        if group:
            if not multivariate:
                raise ValueError(
                    "``group`` option can only be enabled when ``multivariate`` is enabled."
                )
            warnings.warn(
                "``group`` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )
            self._group_decomposed_search_space = _GroupDecomposedSearchSpace(True)

        if constant_liar:
            warnings.warn(
                "``constant_liar`` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

        if constraints_func is not None:
            warnings.warn(
                "The ``constraints_func`` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

        if categorical_distance_func is not None:
            warnings.warn(
                "The ``categorical_distance_func`` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._random_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        if not self._multivariate:
            return {}

        search_space: Dict[str, BaseDistribution] = {}

        if self._group:
            assert self._group_decomposed_search_space is not None
            self._search_space_group = self._group_decomposed_search_space.calculate(study)
            for sub_space in self._search_space_group.search_spaces:
                # Sort keys because Python's string hashing is nondeterministic.
                for name, distribution in sorted(sub_space.items()):
                    if distribution.single():
                        continue
                    search_space[name] = distribution
            return search_space

        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        if self._group:
            assert self._search_space_group is not None
            params = {}
            for sub_space in self._search_space_group.search_spaces:
                search_space = {}
                # Sort keys because Python's string hashing is nondeterministic.
                for name, distribution in sorted(sub_space.items()):
                    if not distribution.single():
                        search_space[name] = distribution
                params.update(self._sample_relative(study, trial, search_space))
            return params
        else:
            return self._sample_relative(study, trial, search_space)

    def _sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE, TrialState.PRUNED)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        # If the number of samples is insufficient, we run random trial.
        if len(trials) < self._n_startup_trials:
            return {}

        return self._sample(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        states = (TrialState.COMPLETE, TrialState.PRUNED)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        # If the number of samples is insufficient, we run random trial.
        if len(trials) < self._n_startup_trials:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        if self._warn_independent_sampling and self._multivariate:
            # Avoid independent warning at the first sampling of `param_name`.
            if any(param_name in trial.params for trial in trials):
                _logger.warning(
                    f"The parameter '{param_name}' in trial#{trial.number} is sampled "
                    "independently instead of being sampled by multivariate TPE sampler. "
                    "(optimization performance may be degraded). "
                    "You can suppress this warning by setting `warn_independent_sampling` "
                    "to `False` in the constructor of `TPESampler`, "
                    "if this independent sampling is intended behavior."
                )

        return self._sample(study, trial, {param_name: param_distribution})[param_name]

    def _get_internal_repr(
        self, trials: list[FrozenTrial], search_space: dict[str, BaseDistribution]
    ) -> dict[str, np.ndarray]:
        values: dict[str, list[float]] = {param_name: [] for param_name in search_space}
        for trial in trials:
            if all((param_name in trial.params) for param_name in search_space):
                for param_name in search_space:
                    param = trial.params[param_name]
                    distribution = trial.distributions[param_name]
                    values[param_name].append(distribution.to_internal_repr(param))
        return {k: np.asarray(v) for k, v in values.items()}

    def _get_parzen_estimators_and_quantiles_for_ctpe(
        self,
        study: Study,
        trials: list[FrozenTrial],
        search_space: dict[str, BaseDistribution],
        n_below_min: int,
    ) -> tuple[list[_ParzenEstimator], list[_ParzenEstimator], list[float]]:
        mpes_good, mpes_bad, quantiles = [], [], []
        feasible_trials_list, infeasible_trials_list = _split_trials_for_constraints(
            trials, n_below_min,
        )
        for feas_trials, infeas_trials in zip(feasible_trials_list, infeasible_trials_list):
            mpes_good.append(self._build_parzen_estimator(study, search_space, feas_trials))
            mpes_bad.append(self._build_parzen_estimator(study, search_space, infeas_trials))
            quantiles.append(len(feas_trials) / max(1, len(feas_trials) + len(infeas_trials)))

        return mpes_good, mpes_bad, quantiles

    def _sample(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        if self._constant_liar:
            states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
        else:
            states = [TrialState.COMPLETE, TrialState.PRUNED]
        use_cache = not self._constant_liar
        trials = study._get_trials(deepcopy=False, states=states, use_cache=use_cache)

        # We divide data into below and above.
        n = sum(trial.state != TrialState.RUNNING for trial in trials)  # Ignore running trials.
        below_trials, above_trials = _split_trials(
            study,
            trials,
            ctpe_default_gamma(n, study, trials, self._gamma) if self._ctpe else self._gamma(n),
            self._constraints_func is not None and not self._ctpe,
        )

        mpes_good = [
            self._build_parzen_estimator(study, search_space, below_trials, handle_below=True)
        ]
        mpes_bad = [self._build_parzen_estimator(study, search_space, above_trials)]
        quantiles = [len(below_trials) / max(1, len(below_trials) + len(above_trials))]
        if self._ctpe:
            _mpes_good, _mpes_bad, _quantiles = self._get_parzen_estimators_and_quantiles_for_ctpe(
                study,
                trials,
                search_space,
                n_below_min=self._gamma(n),
            )
            mpes_good.extend(_mpes_good)
            mpes_bad.extend(_mpes_bad)
            quantiles.extend(_quantiles)

        mpe_good = _ParzenEstimatorList(mpes_good)
        mpe_bad = _ParzenEstimatorList(mpes_bad)

        # NOTE: For c-TPE, len(samples_good) == (n_constraints + 1) * self._n_ei_candidates.
        samples_good = mpe_good.sample(self._rng.rng, self._n_ei_candidates)
        acq_fn_vals = self._compute_acquisition_function(
            samples_good, mpe_good, mpe_bad, quantiles
        )
        ret = TPESampler._compare(samples_good, acq_fn_vals)

        for param_name, dist in search_space.items():
            ret[param_name] = dist.to_external_repr(ret[param_name])

        return ret

    def _build_parzen_estimator(
        self,
        study: Study,
        search_space: dict[str, BaseDistribution],
        trials: list[FrozenTrial],
        handle_below: bool = False,
    ) -> _ParzenEstimator:
        observations = self._get_internal_repr(trials, search_space)
        if handle_below and study._is_multi_objective():
            param_mask_below = []
            for trial in trials:
                param_mask_below.append(
                    all((param_name in trial.params) for param_name in search_space)
                )
            # TODO(nabenabe0928): I do not think this weighting works out for c-TPE.
            weights_below = _calculate_weights_below_for_multi_objective(
                study,
                trials,
                constraints_func=self._constraints_func if not self._ctpe else None,
            )[param_mask_below]
            mpe = _ParzenEstimator(
                observations, search_space, self._parzen_estimator_parameters, weights_below
            )
        else:
            mpe = _ParzenEstimator(observations, search_space, self._parzen_estimator_parameters)

        return mpe

    def _compute_acquisition_function(
        self,
        samples: dict[str, np.ndarray],
        mpe_good: _ParzenEstimatorList,
        mpe_bad: _ParzenEstimatorList,
        quantiles: list[float],
    ) -> np.ndarray:
        log_likelihoods_good = mpe_good.log_pdf(samples)
        log_likelihoods_bad = mpe_bad.log_pdf(samples)
        if not self._ctpe and (len(log_likelihoods_good) != 1 or len(log_likelihoods_bad) != 1):
            raise ValueError(
                "The number of Parzen estimators for below and above in non c-TPE cases "
                f"must be one, but got len(mpe_good)={len(mpe_good)} and "
                f"len(mpe_bad)={len(mpe_bad)}."
            )
        # See: c-TPE: Tree-structured Parzen Estimator with Inequality Constraints for
        # Expensive Hyperparameter Optimization (https://arxiv.org/abs/2211.14411)
        # NOTE: If no constraint exists, acq_fn_vals falls back to the original TPE version.
        _quantiles = np.asarray(quantiles)[:, np.newaxis]
        log_first_term = np.log(_quantiles + EPS)
        log_second_term = (
            np.log(1.0 - _quantiles + EPS) + log_likelihoods_bad - log_likelihoods_good
        )
        acq_fn_vals = np.sum(-np.logaddexp(log_first_term, log_second_term), axis=0)
        return acq_fn_vals

    @classmethod
    def _compare(
        cls,
        samples: Dict[str, np.ndarray],
        acq_fn_vals: np.ndarray,
    ) -> dict[str, int | float]:
        sample_size = next(iter(samples.values())).size
        if sample_size == 0:
            raise ValueError(f"The size of `samples` must be positive, but got {sample_size}.")

        if sample_size != acq_fn_vals.size:
            raise ValueError(
                "The sizes of `samples` and `acq_fn_vals` must be same, but got "
                f"(samples.size, acq_fn_vals.size) = ({sample_size}, {acq_fn_vals.size})."
            )

        best_idx = np.argmax(acq_fn_vals)
        return {k: v[best_idx].item() for k, v in samples.items()}

    @staticmethod
    def hyperopt_parameters() -> Dict[str, Any]:
        """Return the the default parameters of hyperopt (v0.1.2).

        :class:`~optuna.samplers.TPESampler` can be instantiated with the parameters returned
        by this method.

        Example:

            Create a :class:`~optuna.samplers.TPESampler` instance with the default
            parameters of `hyperopt <https://github.com/hyperopt/hyperopt/tree/0.1.2>`_.

            .. testcode::

                import optuna
                from optuna.samplers import TPESampler


                def objective(trial):
                    x = trial.suggest_float("x", -10, 10)
                    return x**2


                sampler = TPESampler(**TPESampler.hyperopt_parameters())
                study = optuna.create_study(sampler=sampler)
                study.optimize(objective, n_trials=10)

        Returns:
            A dictionary containing the default parameters of hyperopt.

        """

        return {
            "consider_prior": True,
            "prior_weight": 1.0,
            "consider_magic_clip": True,
            "consider_endpoints": False,
            "n_startup_trials": 20,
            "n_ei_candidates": 24,
            "gamma": hyperopt_default_gamma,
            "weights": default_weights,
        }

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._random_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        assert state in [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED]
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
        self._random_sampler.after_trial(study, trial, state, values)


def _calculate_nondomination_rank(loss_vals: np.ndarray, n_below: int) -> np.ndarray:
    ranks = np.full(len(loss_vals), -1)
    num_ranked = 0
    rank = 0
    domination_mat = np.all(loss_vals[:, None, :] >= loss_vals[None, :, :], axis=2) & np.any(
        loss_vals[:, None, :] > loss_vals[None, :, :], axis=2
    )
    while num_ranked < n_below:
        counts = np.sum((ranks == -1)[None, :] & domination_mat, axis=1)
        num_ranked += np.sum((counts == 0) & (ranks == -1))
        ranks[(counts == 0) & (ranks == -1)] = rank
        rank += 1
    return ranks


def _split_trials(
    study: Study,
    trials: list[FrozenTrial],
    n_below: int,
    constraints_enabled: bool,
    order_by: Literal["value", "number"] = "number",
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    complete_trials = []
    pruned_trials = []
    running_trials = []
    infeasible_trials = []

    for trial in trials:
        if constraints_enabled and _get_infeasible_trial_score(trial) > 0:
            infeasible_trials.append(trial)
        elif trial.state == TrialState.COMPLETE:
            complete_trials.append(trial)
        elif trial.state == TrialState.PRUNED:
            pruned_trials.append(trial)
        elif trial.state == TrialState.RUNNING:
            running_trials.append(trial)
        else:
            assert False

    # We divide data into below and above.
    below_complete, above_complete = _split_complete_trials(complete_trials, study, n_below)
    n_below -= len(below_complete)
    below_pruned, above_pruned = _split_pruned_trials(pruned_trials, study, n_below)
    n_below -= len(below_pruned)
    below_infeasible, above_infeasible = _split_infeasible_trials(infeasible_trials, n_below)

    below_trials = below_complete + below_pruned + below_infeasible
    above_trials = above_complete + above_pruned + above_infeasible + running_trials
    
    if order_by == "number":
        # This is necessary to weight lower on older trials.
        # Otherwise, trials will be sorted by value. (at least for each state!)
        below_trials.sort(key=lambda trial: trial.number)
        above_trials.sort(key=lambda trial: trial.number)

    return below_trials, above_trials


def _split_complete_trials(
    trials: Sequence[FrozenTrial], study: Study, n_below: int
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    n_below = min(n_below, len(trials))
    if len(study.directions) <= 1:
        return _split_complete_trials_single_objective(trials, study, n_below)
    else:
        return _split_complete_trials_multi_objective(trials, study, n_below)


def _split_complete_trials_single_objective(
    trials: Sequence[FrozenTrial],
    study: Study,
    n_below: int,
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    if study.direction == StudyDirection.MINIMIZE:
        sorted_trials = sorted(trials, key=lambda trial: cast(float, trial.value))
    else:
        sorted_trials = sorted(trials, key=lambda trial: cast(float, trial.value), reverse=True)
    return sorted_trials[:n_below], sorted_trials[n_below:]


def _split_complete_trials_multi_objective(
    trials: Sequence[FrozenTrial],
    study: Study,
    n_below: int,
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    if n_below == 0:
        # The type of trials must be `list`, but not `Sequence`.
        return [], list(trials)

    lvals = np.asarray([trial.values for trial in trials])
    for i, direction in enumerate(study.directions):
        if direction == StudyDirection.MAXIMIZE:
            lvals[:, i] *= -1

    # Solving HSSP for variables number of times is a waste of time.
    nondomination_ranks = _calculate_nondomination_rank(lvals, n_below)
    assert 0 <= n_below <= len(lvals)

    indices = np.array(range(len(lvals)))
    indices_below = np.empty(n_below, dtype=int)

    # Nondomination rank-based selection
    i = 0
    last_idx = 0
    while last_idx < n_below and last_idx + sum(nondomination_ranks == i) <= n_below:
        length = indices[nondomination_ranks == i].shape[0]
        indices_below[last_idx : last_idx + length] = indices[nondomination_ranks == i]
        last_idx += length
        i += 1

    # Hypervolume subset selection problem (HSSP)-based selection
    subset_size = n_below - last_idx
    if subset_size > 0:
        rank_i_lvals = lvals[nondomination_ranks == i]
        rank_i_indices = indices[nondomination_ranks == i]
        worst_point = np.max(rank_i_lvals, axis=0)
        reference_point = np.maximum(1.1 * worst_point, 0.9 * worst_point)
        reference_point[reference_point == 0] = EPS
        selected_indices = _solve_hssp(rank_i_lvals, rank_i_indices, subset_size, reference_point)
        indices_below[last_idx:] = selected_indices

    # NOTE: At least, `below_trials` must be sorted for c-TPE.
    indices_above = np.setdiff1d(np.arange(len(trials)), indices_below)
    below_trials = [trials[i] for i in indices_below]
    above_trials = [trials[i] for i in indices_above]
    return below_trials, above_trials


def _get_pruned_trial_score(trial: FrozenTrial, study: Study) -> tuple[float, float]:
    if len(trial.intermediate_values) > 0:
        step, intermediate_value = max(trial.intermediate_values.items())
        if math.isnan(intermediate_value):
            return -step, float("inf")
        elif study.direction == StudyDirection.MINIMIZE:
            return -step, intermediate_value
        else:
            return -step, -intermediate_value
    else:
        return 1, 0.0


def _split_pruned_trials(
    trials: Sequence[FrozenTrial],
    study: Study,
    n_below: int,
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    n_below = min(n_below, len(trials))
    sorted_trials = sorted(trials, key=lambda trial: _get_pruned_trial_score(trial, study))
    return sorted_trials[:n_below], sorted_trials[n_below:]


def _get_infeasible_trial_score(trial: FrozenTrial) -> float:
    constraint = trial.system_attrs.get(_CONSTRAINTS_KEY)
    if constraint is None:
        warnings.warn(
            f"Trial {trial.number} does not have constraint values."
            " It will be treated as a lower priority than other trials."
        )
        return float("inf")
    else:
        # Violation values of infeasible dimensions are summed up.
        return sum(v for v in constraint if v > 0)


def _split_infeasible_trials(
    trials: Sequence[FrozenTrial], n_below: int
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    n_below = min(n_below, len(trials))
    sorted_trials = sorted(trials, key=_get_infeasible_trial_score)
    return sorted_trials[:n_below], sorted_trials[n_below:]


def _split_trials_for_constraints(
    trials: list[FrozenTrial],
    n_below_min: int,
) -> tuple[list[list[FrozenTrial]], list[list[FrozenTrial]]]:
    n_constraints = _infer_n_constraints(trials)
    if n_constraints == 0:
        warnings.warn("No trials with constraint values were found.")
        return [], []

    indices = np.arange(len(trials))
    cstr_indices = np.array(
        [i for i, t in enumerate(trials) if _CONSTRAINTS_KEY in t.system_attrs]
    )
    # cstr_vals.shape = (n_constraints, len(cstr_indices))
    cstr_vals = np.array([trials[i].system_attrs[_CONSTRAINTS_KEY] for i in cstr_indices]).T
    # Find the n_below_min-th minimum value in each constraint.
    thresholds = np.partition(cstr_vals, kth=n_below_min - 1, axis=-1)[:, n_below_min - 1]
    feasible_trials_list, infeasible_trials_list = [], []
    for threshold, cstr_val in zip(thresholds, cstr_vals):
        # Include at least the indices up to the n_below_min-th min constraint value.
        feasible_indices = cstr_indices[cstr_val <= max(0, threshold)]
        infeasible_indices = np.setdiff1d(indices, feasible_indices)
        feasible_trials_list.append([trials[idx] for idx in feasible_indices])
        infeasible_trials_list.append([trials[idx] for idx in infeasible_indices])

    return feasible_trials_list, infeasible_trials_list


def _calculate_weights_below_for_multi_objective(
    study: Study,
    below_trials: list[FrozenTrial],
    constraints_func: Callable[[FrozenTrial], Sequence[float]] | None,
) -> np.ndarray:
    loss_vals = []
    feasible_mask = np.ones(len(below_trials), dtype=bool)
    for i, trial in enumerate(below_trials):
        # Hypervolume contributions are calculated only using feasible trials.
        if constraints_func is not None:
            if any(constraint > 0 for constraint in constraints_func(trial)):
                feasible_mask[i] = False
                continue
        values = []
        for value, direction in zip(trial.values, study.directions):
            if direction == StudyDirection.MINIMIZE:
                values.append(value)
            else:
                values.append(-value)
        loss_vals.append(values)
    lvals = np.asarray(loss_vals, dtype=float)

    # Calculate weights based on hypervolume contributions.
    n_below = len(lvals)
    weights_below: np.ndarray
    if n_below == 0:
        weights_below = np.asarray([])
    elif n_below == 1:
        weights_below = np.asarray([1.0])
    else:
        worst_point = np.max(lvals, axis=0)
        reference_point = np.maximum(1.1 * worst_point, 0.9 * worst_point)
        reference_point[reference_point == 0] = EPS
        hv = WFG().compute(lvals, reference_point)
        indices_mat = ~np.eye(n_below).astype(bool)
        contributions = np.asarray(
            [hv - WFG().compute(lvals[indices_mat[i]], reference_point) for i in range(n_below)]
        )
        contributions += EPS
        weights_below = np.clip(contributions / np.max(contributions), 0, 1)

    # For now, EPS weight is assigned to infeasible trials.
    weights_below_all = np.full(len(below_trials), EPS)
    weights_below_all[feasible_mask] = weights_below
    return weights_below_all
