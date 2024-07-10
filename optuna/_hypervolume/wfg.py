from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np

from optuna._imports import try_import
from optuna.study._multi_objective import _is_pareto_front


with try_import() as _sortedcontainers_imports:
    import sortedcontainers


def _compute_2d(sorted_pareto_sols: np.ndarray, reference_point: np.ndarray) -> float:
    assert sorted_pareto_sols.shape[1] == 2 and reference_point.shape[0] == 2
    rect_diag_y = np.append(reference_point[1], sorted_pareto_sols[:-1, 1])
    edge_length_x = reference_point[0] - sorted_pareto_sols[:, 0]
    edge_length_y = rect_diag_y - sorted_pareto_sols[:, 1]
    return edge_length_x @ edge_length_y


def _compute_3d(sorted_pareto_sols: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Hypervolume computation algorithm for 3D.
    The time complexity of this algorithm is O(N log N) where N is sorted_pareto_sols.shape[0].

    References:
        Title: Computing and Updating Hypervolume Contributions in Up to Four Dimensions
        Authors: Andreia P. Guerreiro and Carlos M. Fonseca
    """
    # NOTE(nabenabe0928): The indices of X and Y in the sorted lists are the reverse of each other.
    nondominated_X = sortedcontainers.SortedList([-float("inf"), reference_point[1]])
    nondominated_Y = sortedcontainers.SortedList([-float("inf"), reference_point[2]])
    hv = 0.0
    for loss_value in sorted_pareto_sols:
        # Fig. 2 in the paper is easier to understand to track this routine.
        # loss_value is S[10] in the figure.
        # nondominated_X is x of s[5], s[6], s[7], s[8], and s[9]. 
        n_nondominated = len(nondominated_X)
        # nondominated_X[left - 1] < loss_value[1] <= nondominated_X[left]
        # In Fig. 2, nondominated_X is [s[5].x, s[6].x, s[7].x, s[8].x, s[9].x]
        # and s[5].x < s[10].x < s[6].x, so left = 1, i.e. the index of s[6] in nondominated_X.
        left = nondominated_X.bisect_left(loss_value[1])
        # nondominated_Y[- right - 1] < loss_value[2] <= nondominated_Y[-right]    
        # In Fig. 2, nondominated_Y is [s[9].y, s[8].y, s[7].y, s[6].y, s[5].y]
        # and s[9].y < s[10].y < s[8].y, so right = 4.
        right = n_nondominated - nondominated_Y.bisect_left(loss_value[2])
        assert 0 < left <= right < n_nondominated
        # In Fig. 3 (a), the diagonal point of p, i.e. s[10], in the weak-gray rectangular.
        diagonal_point = np.asarray([nondominated_X[right], nondominated_Y[-left]])
        # The surface of weak-gray rectangular including three small dark-gray rectangulars.
        inclusive_hv = np.prod(diagonal_point - loss_value[1:])
        # In Fig. 2, dominated_sols are [s[6], s[7], s[8]].
        dominated_sols = np.stack(
            [nondominated_X[left:right], list(reversed(nondominated_Y[-right:-left]))], axis=-1
        )
        # The surface of weak-gray part excluding three small dark-gray rectangulars.
        # The surface of three small dark-gray rectangulars is calculated by `_compute_2d`.
        surface_increment = inclusive_hv - _compute_2d(dominated_sols, diagonal_point)
        height_from_ref_point = reference_point[0] - loss_value[0]
        hv += surface_increment * height_from_ref_point

        # Delete dominated solutions (s[6], s[7], s[8]) for the next iteration.
        del nondominated_X[left:right]
        del nondominated_Y[-right:-left]
        # Add the current point of interest, i.e. s[10] in Fig. 2, to the nondominated solutions.
        nondominated_X.add(loss_value[1])
        nondominated_Y.add(loss_value[2])

    return hv


def _compute_hv(sorted_loss_vals: np.ndarray, reference_point: np.ndarray) -> float:
    inclusive_hvs = np.prod(reference_point - sorted_loss_vals, axis=-1)
    if inclusive_hvs.shape[0] == 1:
        return float(inclusive_hvs[0])
    elif inclusive_hvs.shape[0] == 2:
        # S(A v B) = S(A) + S(B) - S(A ^ B).
        intersec = np.prod(reference_point - np.maximum(sorted_loss_vals[0], sorted_loss_vals[1]))
        return np.sum(inclusive_hvs) - intersec

    # c.f. Eqs. (6) and (7) of ``A Fast Way of Calculating Exact Hypervolumes``.
    limited_sols_array = np.maximum(sorted_loss_vals[:, np.newaxis], sorted_loss_vals)
    return sum(
        _compute_exclusive_hv(limited_sols_array[i, i + 1 :], inclusive_hv, reference_point)
        for i, inclusive_hv in enumerate(inclusive_hvs)
    )


def _compute_exclusive_hv(
    limited_sols: np.ndarray, inclusive_hv: float, reference_point: np.ndarray
) -> float:
    if limited_sols.shape[0] == 0:
        return inclusive_hv

    on_front = _is_pareto_front(limited_sols, assume_unique_lexsorted=False)
    return inclusive_hv - _compute_hv(limited_sols[on_front], reference_point)


def compute_hypervolume(
    loss_vals: np.ndarray, reference_point: np.ndarray, assume_pareto: bool = False
) -> float:
    """Hypervolume calculator for any dimension.

    This class exactly calculates the hypervolume for any dimension.
    For 3 dimensions or higher, the WFG algorithm will be used.
    Please refer to ``A Fast Way of Calculating Exact Hypervolumes`` for the WFG algorithm.

    .. note::
        This class is used for computing the hypervolumes of points in multi-objective space.
        Each coordinate of each point represents a ``values`` of the multi-objective function.

    .. note::
        We check that each objective is to be minimized. Transform objective values that are
        to be maximized before calling this class's ``compute`` method.

    Args:
        loss_vals:
            An array of loss value vectors to calculate the hypervolume.
        reference_point:
            The reference point used to calculate the hypervolume.
        assume_pareto:
            Whether to assume the Pareto optimality to ``loss_vals``.
            In other words, if ``True``, none of loss vectors are dominated by another.

    Returns:
        The hypervolume of the given arguments.

    """

    if not np.all(loss_vals <= reference_point):
        raise ValueError(
            "All points must dominate or equal the reference point. "
            "That is, for all points in the loss_vals and the coordinate `i`, "
            "`loss_vals[i] <= reference_point[i]`."
        )
    if not np.all(np.isfinite(reference_point)):
        # reference_point does not have nan, thanks to the verification above.
        return float("inf")

    if not assume_pareto:
        unique_lexsorted_loss_vals = np.unique(loss_vals, axis=0)
        on_front = _is_pareto_front(unique_lexsorted_loss_vals, assume_unique_lexsorted=True)
        sorted_pareto_sols = unique_lexsorted_loss_vals[on_front]
    else:
        sorted_pareto_sols = loss_vals[loss_vals[:, 0].argsort()]

    if reference_point.shape[0] == 2:
        return _compute_2d(sorted_pareto_sols, reference_point)
    if reference_point.shape[0] == 3:
        if _sortedcontainers_imports.is_successful():
            return _compute_3d(sorted_pareto_sols, reference_point)
        else:
            warnings.warn(
                "TPESampler for 3-objective problems will be quicker with ``sortedcontainers``."
            )

    return _compute_hv(sorted_pareto_sols, reference_point)
