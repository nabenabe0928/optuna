from __future__ import annotations

import numpy as np

import optuna


def _solve_hssp_2d(
    sorted_pareto_sols: np.ndarray, subset_size: int, reference_point: np.ndarray
) -> np.ndarray:
    # This function can be used for sorted_pareto_sols as well.
    # The time complexity is O(subset_size * rank_i_loss_vals.shape[0]).
    assert sorted_pareto_sols.shape[-1] == 2 and subset_size <= sorted_pareto_sols.shape[0]
    n_trials = sorted_pareto_sols.shape[0]
    # rank_i_loss_vals is unique-lexsorted in solve_hssp.
    sorted_indices = np.arange(sorted_pareto_sols.shape[0])
    # The diagonal points for each rectangular to calculate the hypervolume contributions.
    rect_diags = np.repeat(reference_point[np.newaxis, :], n_trials, axis=0)
    selected_indices = np.zeros(subset_size, dtype=int)
    for i in range(subset_size):
        contribs = np.prod(rect_diags - sorted_pareto_sols[sorted_indices], axis=-1)
        max_index = np.argmax(contribs)
        selected_indices[i] = sorted_indices[max_index]
        # Remove the chosen point.
        sorted_indices = sorted_indices[keep := sorted_indices != selected_indices[i]]
        # Update the diagonal points for each hypervolume contribution calculation.
        rect_diags = np.minimum(sorted_pareto_sols[selected_indices[i]], rect_diags[keep])

    return selected_indices


def _lazy_contribs_update(
    contribs: np.ndarray,
    pareto_loss_values: np.ndarray,
    selected_vecs: np.ndarray,
    reference_point: np.ndarray,
    hv_selected: float,
) -> np.ndarray:
    """Lazy update the hypervolume contributions.

    S=selected_indices - {indices[max_index]}, T=selected_indices, and S' is a subset of S.
    As we would like to know argmax H(T v {i}) in the next iteration, we can skip HV
    calculations for j if H(T v {i}) - H(T) > H(S' v {j}) - H(S') >= H(T v {j}) - H(T).
    We used the submodularity for the inequality above. As the upper bound of contribs[i] is
    H(S' v {j}) - H(S'), we start to update from i with a higher upper bound so that we can
    skip more HV calculations.
    """

    # The HV difference only using the latest selected point and a candidate is a simple, yet
    # obvious, contribution upper bound. Denote t as the latest selected index and j as an
    # unselected index. Then, H(T v {j}) - H(T) <= H({t} v {j}) - H({t}) holds where the inequality
    # comes from submodularity. We use the inclusion-exclusion principle to calculate the RHS.
    single_volume = np.prod(reference_point - pareto_loss_values, axis=1)
    intersection = np.maximum(selected_vecs[-1], pareto_loss_values)
    intersection_volume = np.prod(reference_point - intersection, axis=1)
    contribs = np.minimum(contribs, single_volume - intersection_volume)

    max_contrib = 0.0
    n_objectives = reference_point.size
    index_from_larger_upper_bound_contrib = np.argsort(-contribs)
    new_selected_vecs = np.concatenate([selected_vecs, np.empty((1, n_objectives))], axis=0)
    for i in index_from_larger_upper_bound_contrib:
        if contribs[i] < max_contrib:
            # Lazy evaluation to reduce HV calculations.
            # If contribs[i] will not be the maximum next, it is unnecessary to compute it.
            continue

        new_selected_vecs[-1] = pareto_loss_values[i]
        hv_plus = optuna._hypervolume.compute_hypervolume(
            new_selected_vecs, reference_point, assume_pareto=True
        )
        # inf - inf in the contribution calculation is always inf.
        contribs[i] = hv_plus - hv_selected if not np.isinf(hv_plus) else np.inf
        max_contrib = max(contribs[i], max_contrib)

    return contribs


def _solve_hssp_on_unique_lexsorted_pareto_sols(
    sorted_pareto_sols: np.ndarray, subset_size: int, reference_point: np.ndarray
) -> np.ndarray:
    if not np.isfinite(reference_point).all() or len(sorted_pareto_sols) == subset_size:
        return np.arange(subset_size)
    if sorted_pareto_sols.shape[-1] == 2:
        return _solve_hssp_2d(sorted_pareto_sols, subset_size, reference_point)

    assert subset_size < len(sorted_pareto_sols)
    (n_solutions, n_objectives) = sorted_pareto_sols.shape
    # The following logic can be used for non-unique sorted_pareto_sols as well.
    contribs = np.prod(reference_point - sorted_pareto_sols, axis=-1)
    selected_indices = []
    remaining_indices = np.arange(n_solutions)
    hv = 0
    for k in range(subset_size):
        max_index = np.argmax(contribs).item()
        selected_indices.append(selected_index := remaining_indices[max_index])
        hv += contribs[max_index]
        remaining_indices = remaining_indices[keep := remaining_indices != selected_index]
        if k == subset_size - 1:  # We do not need to update contribs at the last iteration.
            break

        contribs = _lazy_contribs_update(
            contribs[keep],
            sorted_pareto_sols[remaining_indices],
            sorted_pareto_sols[selected_indices],
            reference_point,
            hv,
        )
    return np.asarray(selected_indices, dtype=int)


def _solve_hssp(
    rank_i_loss_vals: np.ndarray,
    rank_i_indices: np.ndarray,
    subset_size: int,
    reference_point: np.ndarray,
) -> np.ndarray:
    """Solve a hypervolume subset selection problem (HSSP) via a greedy algorithm.

    This method is a 1-1/e approximation algorithm to solve HSSP.

    For further information about algorithms to solve HSSP, please refer to the following
    paper:

    - `Greedy Hypervolume Subset Selection in Low Dimensions
       <https://doi.org/10.1162/EVCO_a_00188>`__
    """
    if subset_size == rank_i_indices.size:
        return rank_i_indices

    rank_i_unique_loss_vals, indices_of_unique_loss_vals = np.unique(
        rank_i_loss_vals, return_index=True, axis=0
    )
    n_unique = indices_of_unique_loss_vals.size
    if n_unique < subset_size:
        chosen = np.zeros(rank_i_indices.size, dtype=bool)
        chosen[indices_of_unique_loss_vals] = True
        duplicated_indices = np.arange(rank_i_indices.size)[~chosen]
        chosen[duplicated_indices[: subset_size - n_unique]] = True
        return rank_i_indices[chosen]

    selected_indices = _solve_hssp_on_unique_lexsorted_pareto_sols(
        rank_i_unique_loss_vals, subset_size, reference_point
    )
    return rank_i_indices[indices_of_unique_loss_vals[selected_indices]]
