from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import cast
from typing import TYPE_CHECKING
import warnings

import numpy as np

import optuna
from optuna._experimental import experimental_class
from optuna._experimental import warn_experimental_argument
from optuna.distributions import BaseDistribution
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import StudyDirection
from optuna.study._multi_objective import _is_pareto_front
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    import torch

    import optuna._gp.acqf as acqf_module
    import optuna._gp.gp as gp
    import optuna._gp.optim_mixed as optim_mixed
    import optuna._gp.prior as prior
    import optuna._gp.search_space as gp_search_space
    from optuna.study import Study
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")
    gp_search_space = _LazyImport("optuna._gp.search_space")
    gp = _LazyImport("optuna._gp.gp")
    optim_mixed = _LazyImport("optuna._gp.optim_mixed")
    acqf_module = _LazyImport("optuna._gp.acqf")
    prior = _LazyImport("optuna._gp.prior")


EPS = 1e-10


@experimental_class("3.6.0")
class GPSampler(BaseSampler):
    """Sampler using Gaussian process-based Bayesian optimization.

    This sampler fits a Gaussian process (GP) to the objective function and optimizes
    the acquisition function to suggest the next parameters.

    The current implementation uses:
        - Matern kernel with nu=2.5 (twice differentiable),
        - Automatic relevance determination (ARD) for the length scale of each parameter,
        - Gamma prior for inverse squared lengthscales, kernel scale, and noise variance,
        - Log Expected Improvement (logEI) as the acquisition function, and
        - Quasi-Monte Carlo (QMC) sampling to optimize the acquisition function.

    .. note::
        This sampler requires ``scipy`` and ``torch``.
        You can install these dependencies with ``pip install scipy torch``.

    Args:
        seed:
            Random seed to initialize internal random number generator.
            Defaults to :obj:`None` (a seed is picked randomly).

        independent_sampler:
            Sampler used for initial sampling (for the first ``n_startup_trials`` trials)
            and for conditional parameters. Defaults to :obj:`None`
            (a random sampler with the same ``seed`` is used).

        n_startup_trials:
            Number of initial trials. Defaults to 10.

        deterministic_objective:
            Whether the objective function is deterministic or not.
            If :obj:`True`, the sampler will fix the noise variance of the surrogate model to
            the minimum value (slightly above 0 to ensure numerical stability).
            Defaults to :obj:`False`.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        n_startup_trials: int = 10,
        deterministic_objective: bool = False,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
    ) -> None:
        self._rng = LazyRandomState(seed)
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._intersection_search_space = optuna.search_space.IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials
        self._log_prior: "Callable[[gp.KernelParamsTensor], torch.Tensor]" = (
            prior.default_log_prior
        )
        self._minimum_noise: float = prior.DEFAULT_MINIMUM_NOISE_VAR
        # We cache the kernel parameters for initial values of fitting the next time.
        self._objectives_kernel_params_cache: "list[gp.KernelParamsTensor] | None" = None
        self._constraints_kernel_params_cache: "list[gp.KernelParamsTensor] | None" = None
        self._optimize_n_samples: int = 2048
        self._deterministic = deterministic_objective
        self._constraints_func = constraints_func

        if constraints_func is not None:
            warn_experimental_argument("constraints_func")

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space = {}
        for name, distribution in self._intersection_search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

    def _optimize_acqf(
        self, acqf: "acqf_module.BaseAcquisitionFunc", best_params: np.ndarray | None
    ) -> np.ndarray:
        # Advanced users can override this method to change the optimization algorithm.
        # However, we do not make any effort to keep backward compatibility between versions.
        # Particularly, we may remove this function in future refactoring.
        normalized_params, _acqf_val = optim_mixed.optimize_acqf_mixed(
            acqf,
            warmstart_normalized_params_array=best_params,
            n_preliminary_samples=2048,
            n_local_search=10,
            tol=1e-4,
            rng=self._rng.rng,
        )
        return normalized_params

    def _get_constrained_acqf(
        self,
        objective_acqf: acqf_module.LogEI | acqf_module.LogEHVI,
        constraint_vals: np.ndarray,
        internal_search_space: gp_search_space.SearchSpace,
        normalized_params: np.ndarray,
    ) -> acqf_module.ConstrainedLogEI:
        constraint_vals = _warn_and_convert_inf(constraint_vals)
        means = np.mean(constraint_vals, axis=0)
        stds = np.std(constraint_vals, axis=0)
        standardized_constraint_vals = (constraint_vals - means) / np.maximum(EPS, stds)
        if self._objectives_kernel_params_cache is not None and len(
            self._objectives_kernel_params_cache[0].inverse_squared_lengthscales
        ) != len(internal_search_space.scale_types):
            # Clear cache if the search space changes.
            self._constraints_kernel_params_cache = None

        is_categorical = internal_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
        constraints_kernel_params = []
        constraints_acqf_params = []
        constraint_thresholds = []
        for i, (vals, mean, std) in enumerate(zip(standardized_constraint_vals.T, means, stds)):
            cache = (
                self._constraints_kernel_params_cache and self._constraints_kernel_params_cache[i]
            )
            assert isinstance(cache, gp.KernelParamsTensor) or cache is None

            kernel_params = gp.fit_kernel_params(
                X=normalized_params,
                Y=vals,
                is_categorical=is_categorical,
                log_prior=self._log_prior,
                minimum_noise=self._minimum_noise,
                initial_kernel_params=cache,
                deterministic_objective=self._deterministic,
            )
            constraints_kernel_params.append(kernel_params)

            # Since 0 is the threshold value, we use the normalized value of 0.
            constraint_thresholds.append(-mean / max(EPS, std))

        self._constraints_kernel_params_cache = constraints_kernel_params

        return ConstrainedLogEI(
            constraint_kernel_params_list=constraints_kernel_params,
            X=normalized_params,
            constraint_vals=standardized_constraint_vals,
            search_space=internal_search_space,
            objective_acqf=objective_acqf,
            constraint_thresholds=constraint_thresholds,
        )

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        if len(trials) < self._n_startup_trials:
            return {}

        (
            internal_search_space,
            normalized_params,
        ) = gp_search_space.get_search_space_and_normalized_params(trials, search_space)

        _signs = np.array([-1 if d == StudyDirection.MINIMIZE else 1 for d in study.directions])
        score_vals = _signs * _warn_and_convert_inf(
            np.array([cast(list, trial.values) for trial in trials])
        )
        standardized_score_vals = (
            (score_vals - np.mean(score_vals, axis=0)) / np.maximum(EPS, np.std(score_vals, axis=0))
        )

        if self._objectives_kernel_params_cache is not None and len(
            self._objectives_kernel_params_cache[0].inverse_squared_lengthscales
        ) != len(internal_search_space.scale_types):
            # Clear cache if the search space changes.
            self._objectives_kernel_params_cache = None

        objectives_kernel_params = []
        is_categorical = internal_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
        for i, svs in enumerate(standardized_score_vals.T):
            cache = (
                self._objectives_kernel_params_cache and self._objectives_kernel_params_cache[i]
            )
            assert isinstance(cache, gp.KernelParamsTensor) or cache is None
            kernel_params = gp.fit_kernel_params(
                X=normalized_params,
                Y=svs,
                is_categorical=is_categorical,
                log_prior=self._log_prior,
                minimum_noise=self._minimum_noise,
                initial_kernel_params=cache,
                deterministic_objective=self._deterministic,
            )
            objectives_kernel_params.append(kernel_params)

        self._objectives_kernel_params_cache = objectives_kernel_params
        if study._is_multi_objective():
            acqf = acqf_module.LogEHVI(
                objective_kernel_params_list=objectives_kernel_params,
                search_space=internal_search_space,
                X=normalized_params,
                Y=standardized_score_vals,
                seed=self._rng.rng.randint(1 << 31),
            )
        else:
            acqf = acqf_module.LogEI(
                kernel_params=objectives_kernel_params[0],
                search_space=internal_search_space,
                X=normalized_params,
                Y=standardized_score_vals[:, 0],
            )

        best_params = normalized_params[
            _is_pareto_front(-standardized_score_vals, assume_unique_lexsorted=False)
        ]
        if self._constraints_func is not None:
            raise NotImplementedError("Not available.")
            constraint_vals, is_feasible = _get_constraint_vals_and_feasibility(study, trials)
            is_all_infeasible = not np.any(is_feasible)

            # TODO(kAIto47802): If is_all_infeasible, the acquisition function for the objective
            # function is ignored, so skipping the computation of kernel_params and acqf_params
            # can improve speed.
            # TODO(kAIto47802): Consider the case where all trials are feasible. We can ignore
            # constraints in this case.
            threshold = (
                -np.inf if is_all_infeasible else np.max(standardized_score_vals[is_feasible])
            )
            objective_acqf = acqf_module.LogEI(
                kernel_params=kernel_params,
                search_space=internal_search_space,
                X=normalized_params,
                Y=standardized_score_vals,
                threshold=threshold,
            )
            acqf = self._get_constrained_acqf(
                objective_acqf, constraint_vals, internal_search_space, normalized_params
            )
            best_params = (
                None
                if is_all_infeasible
                else normalized_params[np.argmax(standardized_score_vals[is_feasible]), :]
            )

        normalized_param = self._optimize_acqf(acqf, best_params)
        return gp_search_space.get_unnormalized_param(search_space, normalized_param)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        # self._raise_error_if_multi_objective(study)
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
        self._independent_sampler.after_trial(study, trial, state, values)


def _warn_and_convert_inf(
    values: np.ndarray,
) -> np.ndarray:
    if np.any(~np.isfinite(values)):
        warnings.warn(
            "GPSampler cannot handle +/-inf, so we clip them to the best/worst finite value."
        )

        finite_vals_with_nan = np.where(np.isfinite(values), values, np.nan)
        is_any_finite = np.any(np.isfinite(finite_vals_with_nan), axis=0)
        best_finite_vals = np.where(is_any_finite, np.nanmax(finite_vals_with_nan, axis=0), 0.0)
        worst_finite_vals = np.where(is_any_finite, np.nanmin(finite_vals_with_nan, axis=0), 0.0)
        return np.clip(values, worst_finite_vals, best_finite_vals)

    return values


def _get_constraint_vals_and_feasibility(
    study: Study, trials: list[FrozenTrial]
) -> tuple[np.ndarray, np.ndarray]:
    _constraint_vals = [
        study._storage.get_trial_system_attrs(trial._trial_id).get(_CONSTRAINTS_KEY, ())
        for trial in trials
    ]
    if any(len(_constraint_vals[0]) != len(c) for c in _constraint_vals):
        raise ValueError("The number of constraints must be the same for all trials.")

    constraint_vals = np.array(_constraint_vals)
    assert len(constraint_vals.shape) == 2, "constraint_vals must be a 2d array."
    is_feasible = np.all(constraint_vals <= 0, axis=1)
    assert not isinstance(is_feasible, np.bool_), "MyPy Redefinition for NumPy v2.2.0."
    return constraint_vals, is_feasible
