from __future__ import annotations

from collections import Callable
from typing import Sequence, Literal

import optuna


def run_study(
    objective: Callable[[optuna.Trial], float | Sequence[float]],
    constraints_func: Callable[[optuna.trial.FrozenTrial], Sequence[float]],
    seed: int,
    gamma_type: str,
    ctpe: bool,
    study_name: str,
    storage: str,
    directions: list[Literal["minimize", "maximize"]],
    n_trials: int = 100,
) -> None:
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        constraints_func=constraints_func,
        ctpe=ctpe,
        seed=seed,
        gamma=(
            optuna.samplers._tpe.sampler.default_gamma if gamma_type == "linear"
            else optuna.samplers._tpe.sampler.hyperopt_default_gamma
        )
    )
    kwargs = dict(directions=directions) if len(directions) > 1 else dict(direction=directions[0])
    study = optuna.create_study(sampler=sampler, storage=storage, study_name=study_name, **kwargs)
    study.optimize(objective, n_trials=n_trials)
