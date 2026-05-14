from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

import optuna
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study._constrained_optimization import _OUTCOME_CONSTRAINTS_KEY
from optuna.study._constrained_optimization import _OUTCOME_CONSTRAINT_OPS_KEY
from optuna.trial import FrozenTrial


def test_basic_leq_constraint() -> None:
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        trial.report_outcome_constraint("c1", lhs=x, rhs=5.0)
        return x**2

    study = optuna.create_study()
    study.optimize(objective, n_trials=3)

    for trial in study.trials:
        assert _CONSTRAINTS_KEY in trial.system_attrs
        raw = trial.system_attrs[_OUTCOME_CONSTRAINTS_KEY]
        assert "c1" in raw
        expected = raw["c1"]["lhs"] - raw["c1"]["rhs"]
        assert trial.system_attrs[_CONSTRAINTS_KEY] == (expected,)


def test_basic_geq_constraint() -> None:
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        trial.report_outcome_constraint("c1", lhs=x, rhs=2.0, op=">=")
        return x**2

    study = optuna.create_study()
    study.optimize(objective, n_trials=3)

    for trial in study.trials:
        raw = trial.system_attrs[_OUTCOME_CONSTRAINTS_KEY]
        expected = raw["c1"]["rhs"] - raw["c1"]["lhs"]
        assert trial.system_attrs[_CONSTRAINTS_KEY] == (expected,)


def test_multiple_constraints() -> None:
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        trial.report_outcome_constraint("a_bound", lhs=x, rhs=5.0)
        trial.report_outcome_constraint("b_bound", lhs=y, rhs=3.0, op=">=")
        return x**2 + y**2

    study = optuna.create_study()
    study.optimize(objective, n_trials=3)

    for trial in study.trials:
        constraints = trial.system_attrs[_CONSTRAINTS_KEY]
        raw = trial.system_attrs[_OUTCOME_CONSTRAINTS_KEY]
        assert len(constraints) == 2
        # Alphabetical order: a_bound (<=), b_bound (>=)
        assert constraints[0] == raw["a_bound"]["lhs"] - raw["a_bound"]["rhs"]
        assert constraints[1] == raw["b_bound"]["rhs"] - raw["b_bound"]["lhs"]


def test_rhs_varies_across_trials() -> None:
    thresholds = [5.0, 3.0, 7.0]

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        rhs = thresholds[trial.number]
        trial.report_outcome_constraint("c1", lhs=x, rhs=rhs)
        return x**2

    study = optuna.create_study()
    study.optimize(objective, n_trials=3)

    for i, trial in enumerate(study.trials):
        raw = trial.system_attrs[_OUTCOME_CONSTRAINTS_KEY]
        assert raw["c1"]["rhs"] == thresholds[i]


def test_op_consistency_error() -> None:
    call_count = 0

    def objective(trial: optuna.Trial) -> float:
        nonlocal call_count
        x = trial.suggest_float("x", -10, 10)
        op = "<=" if call_count == 0 else ">="
        trial.report_outcome_constraint("c1", lhs=x, rhs=5.0, op=op)
        call_count += 1
        return x**2

    study = optuna.create_study()
    study.optimize(objective, n_trials=1)
    with pytest.raises(ValueError, match="previously registered"):
        study.optimize(objective, n_trials=1)


def test_invalid_op() -> None:
    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(ValueError, match="op must be"):
        trial.report_outcome_constraint("c1", lhs=1.0, rhs=2.0, op="<")


def test_empty_constraint_name() -> None:
    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(ValueError, match="non-empty"):
        trial.report_outcome_constraint("", lhs=1.0, rhs=2.0)


def test_nan_lhs() -> None:
    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(ValueError, match="NaN"):
        trial.report_outcome_constraint("c1", lhs=float("nan"), rhs=2.0)


def test_nan_rhs() -> None:
    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(ValueError, match="NaN"):
        trial.report_outcome_constraint("c1", lhs=1.0, rhs=float("nan"))


def test_invalid_lhs_type() -> None:
    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(TypeError, match="lhs"):
        trial.report_outcome_constraint("c1", lhs="abc", rhs=2.0)  # type: ignore


def test_invalid_rhs_type() -> None:
    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(TypeError, match="rhs"):
        trial.report_outcome_constraint("c1", lhs=1.0, rhs="abc")  # type: ignore


def test_conflict_with_sampler_constraints_func() -> None:
    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        return [trial.params["x"] - 5.0]

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        trial.report_outcome_constraint("c1", lhs=x, rhs=5.0)
        return x**2

    sampler = optuna.samplers.TPESampler(constraints_func=constraints_func)
    study = optuna.create_study(sampler=sampler)
    with pytest.raises(ValueError, match="Cannot use both"):
        study.optimize(objective, n_trials=1)


def test_ask_and_tell() -> None:
    study = optuna.create_study()
    trial = study.ask()
    x = 3.0
    trial.report_outcome_constraint("c1", lhs=x, rhs=5.0)
    study.tell(trial, x**2)

    frozen = study.trials[0]
    assert _CONSTRAINTS_KEY in frozen.system_attrs
    assert frozen.system_attrs[_CONSTRAINTS_KEY] == (3.0 - 5.0,)


def test_feasibility_with_best_trial() -> None:
    values = [1.0, 10.0, 5.0]
    constraint_lhs = [100.0, 1.0, 1.0]  # First trial infeasible

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0, 1)
        idx = trial.number
        trial.report_outcome_constraint("c1", lhs=constraint_lhs[idx], rhs=5.0)
        return values[idx]

    study = optuna.create_study()
    study.optimize(objective, n_trials=3)

    # Best feasible trial should be trial 2 (value=5.0) or trial 1 (value=10.0),
    # but not trial 0 (value=1.0) which is infeasible.
    # Actually for minimize, trial 2 has value 5.0 and trial 1 has value 10.0.
    # Trial 0 has value 1.0 but is infeasible (lhs=100 > rhs=5).
    # So best feasible is trial 2 with value 5.0.
    assert study.best_trial.number == 2
    assert study.best_trial.values == [5.0]


def test_alphabetical_ordering() -> None:
    def objective(trial: optuna.Trial) -> float:
        trial.report_outcome_constraint("z_last", lhs=1.0, rhs=2.0)
        trial.report_outcome_constraint("a_first", lhs=3.0, rhs=4.0)
        trial.report_outcome_constraint("m_middle", lhs=5.0, rhs=6.0, op=">=")
        return 0.0

    study = optuna.create_study()
    study.optimize(objective, n_trials=1)

    constraints = study.trials[0].system_attrs[_CONSTRAINTS_KEY]
    # a_first (<=): 3 - 4 = -1
    # m_middle (>=): 6 - 5 = 1
    # z_last (<=): 1 - 2 = -1
    assert constraints == (-1.0, 1.0, -1.0)


def test_op_registry_stored_in_study() -> None:
    def objective(trial: optuna.Trial) -> float:
        trial.report_outcome_constraint("c1", lhs=1.0, rhs=2.0)
        trial.report_outcome_constraint("c2", lhs=3.0, rhs=4.0, op=">=")
        return 0.0

    study = optuna.create_study()
    study.optimize(objective, n_trials=1)

    study_attrs = study._storage.get_study_system_attrs(study._study_id)
    ops = study_attrs[_OUTCOME_CONSTRAINT_OPS_KEY]
    assert ops == {"c1": "<=", "c2": ">="}


def test_fixed_trial_noop() -> None:
    trial = optuna.trial.FixedTrial({"x": 1.0})
    trial.report_outcome_constraint("c1", lhs=1.0, rhs=2.0)
