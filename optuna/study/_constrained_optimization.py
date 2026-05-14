from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial


_CONSTRAINTS_KEY = "constraints"
_OUTCOME_CONSTRAINTS_KEY = "outcome_constraints"
_OUTCOME_CONSTRAINT_OPS_KEY = "outcome_constraint_ops"


def _get_feasible_trials(trials: Sequence[FrozenTrial]) -> list[FrozenTrial]:
    """Return feasible trials from given trials.

    This function assumes that the trials were created in constrained optimization.
    Therefore, if there is no violation value in the trial, it is considered infeasible.


    Returns:
        A list of feasible trials.
    """

    feasible_trials = []
    for trial in trials:
        constraints = trial.system_attrs.get(_CONSTRAINTS_KEY)
        if constraints is not None and all(x <= 0.0 for x in constraints):
            feasible_trials.append(trial)
    return feasible_trials


def _has_constraints(trials: Sequence[FrozenTrial]) -> bool:
    return any(_CONSTRAINTS_KEY in t.system_attrs for t in trials)


def _process_outcome_constraints(
    study: Study,
    frozen_trial: FrozenTrial,
    state: TrialState,
) -> None:
    if state not in (TrialState.COMPLETE, TrialState.PRUNED):
        return

    raw = frozen_trial.system_attrs.get(_OUTCOME_CONSTRAINTS_KEY)
    if not raw:
        return

    ops = study._storage.get_study_system_attrs(study._study_id).get(
        _OUTCOME_CONSTRAINT_OPS_KEY, {}
    )

    normalized: list[float] = []
    for name in sorted(raw):
        entry = raw[name]
        lhs = entry["lhs"]
        rhs = entry["rhs"]
        op = ops.get(name, entry.get("op", "<="))

        if math.isnan(lhs) or math.isnan(rhs):
            raise ValueError(
                f"Outcome constraint '{name}' has NaN value (lhs={lhs}, rhs={rhs})."
            )

        if op == "<=":
            normalized.append(lhs - rhs)
        else:
            normalized.append(rhs - lhs)

    study._storage.set_trial_system_attr(
        frozen_trial._trial_id, _CONSTRAINTS_KEY, tuple(normalized)
    )
