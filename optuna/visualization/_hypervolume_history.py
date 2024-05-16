from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from optuna._experimental import experimental_func
from optuna._hypervolume import WFG
from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study import Study
from optuna.study._multi_objective import _fast_non_domination_rank
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


class _HypervolumeHistoryInfo(NamedTuple):
    trial_numbers: list[int]
    values: list[float]


@experimental_func("3.3.0")
def plot_hypervolume_history(
    study: Study,
    reference_point: Sequence[float],
) -> "go.Figure":
    """Plot hypervolume history of all trials in a study.

    Example:

        The following code snippet shows how to plot optimization history.

        .. plotly::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", 0, 5)
                y = trial.suggest_float("y", 0, 3)

                v0 = 4 * x ** 2 + 4 * y ** 2
                v1 = (x - 5) ** 2 + (y - 5) ** 2
                return v0, v1


            study = optuna.create_study(directions=["minimize", "minimize"])
            study.optimize(objective, n_trials=50)

            reference_point=[100., 50.]
            fig = optuna.visualization.plot_hypervolume_history(study, reference_point)
            fig.show()

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their hypervolumes.
            The number of objectives must be 2 or more.

        reference_point:
            A reference point to use for hypervolume computation.
            The dimension of the reference point must be the same as the number of objectives.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.
    """

    _imports.check()

    if not study._is_multi_objective():
        raise ValueError(
            "Study must be multi-objective. For single-objective optimization, "
            "please use plot_optimization_history instead."
        )

    if len(reference_point) != len(study.directions):
        raise ValueError(
            "The dimension of the reference point must be the same as the number of objectives."
        )

    info = _get_hypervolume_history_info(study, np.asarray(reference_point, dtype=np.float64))
    return _get_hypervolume_history_plot(info)


def _get_hypervolume_history_plot(
    info: _HypervolumeHistoryInfo,
) -> "go.Figure":
    layout = go.Layout(
        title="Hypervolume History Plot",
        xaxis={"title": "Trial"},
        yaxis={"title": "Hypervolume"},
    )

    data = go.Scatter(
        x=info.trial_numbers,
        y=info.values,
        mode="lines+markers",
    )
    return go.Figure(data=data, layout=layout)


def _get_loss_values_and_non_dominated_rank(
    completed_trials: list[FrozenTrial], signs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    
    objective_values = []
    penalty = []
    for t in completed_trials:
        objective_values.append(t.values)
        if _CONSTRAINTS_KEY not in t.system_attrs:
            penalty.append(np.nan)
            continue
        
        constraint_values = t.system_attrs[_CONSTRAINTS_KEY]
        penalty.append(sum([max(0, v) for v in constraint_values]))

    consider_constraint = not np.all(np.isnan(penalty))
    loss_values = signs * np.asarray(objective_values)
    if consider_constraint:
        penalty_array = np.asarray(penalty)
        nd_ranks = _fast_non_domination_rank(loss_values, penalty_array)
        is_infeasible = penalty_array > 0
        nd_ranks[is_infeasible] = -1
    else:
        nd_ranks = _fast_non_domination_rank(loss_values)

    return loss_values, nd_ranks


def _get_hypervolume_history_info(
    study: Study, reference_point: np.ndarray
) -> _HypervolumeHistoryInfo:
    completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

    if len(completed_trials) == 0:
        _logger.warning("Your study does not have any completed trials.")

    # Our hypervolume computation module assumes that all objectives are minimized.
    # Here we transform the objective values and the reference point.
    signs = np.asarray([1 if d == StudyDirection.MINIMIZE else -1 for d in study.directions])
    minimization_ref_point = signs * reference_point
    loss_vals, nd_ranks = _get_loss_values_and_non_dominated_rank(completed_trials, signs)

    # Only feasible trials are considered in hypervolume computation.
    best_hypervolume = 0.0
    best_nd_rank = 1 << 30  # Any big integer.
    values = []
    cur_pareto_indices = []
    is_values_valid = np.all(loss_vals <= minimization_ref_point, axis=-1)
    for i in range(len(completed_trials)):
        if nd_ranks[i] == -1 or not is_values_valid[i]:
            # Infeasible or the current best trials do not dominate the reference point.
            values.append(best_hypervolume)
            continue

        cur_loss_vals_is_better = loss_vals[i] <= loss_vals[cur_pareto_indices]
        if nd_ranks[i] <= best_nd_rank or not np.any(np.all(~cur_loss_vals_is_better, axis=-1)):
            dominated = np.all(cur_loss_vals_is_better, axis=-1)
            cur_pareto_indices = np.array(cur_pareto_indices)[~dominated].tolist()
            cur_pareto_indices.append(i)
            best_nd_rank = min(nd_ranks[i], best_nd_rank)
            best_hypervolume = WFG().compute(loss_vals[cur_pareto_indices], minimization_ref_point)

        values.append(best_hypervolume)

    if not np.any(nd_ranks == 0):
        _logger.warning("Your study does not have any feasible trials.")

    trial_numbers = [t.number for t in completed_trials]
    return _HypervolumeHistoryInfo(trial_numbers, values)
