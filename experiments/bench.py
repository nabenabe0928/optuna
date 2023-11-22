from __future__ import annotations

import os
from argparse import ArgumentParser

from chpobench import HPOBench, HPOLib, JAHSBench201

from experiments.runner import run_study

import optuna


_CONSTRAINT_KEY = "constraints"


def objective(trial: optuna.Trial, bench) -> float:
    eval_config = {}
    for name, dist in bench.config_space.items():
        if hasattr(dist, "choices"):
            eval_config[name] = trial.suggest_categorical(name=name, choices=dist.choices)
        elif hasattr(dist, "seq"):
            seq = dist.seq
            idx = trial.suggest_int(name=f"{name}-index", low=0, high=len(seq) - 1)
            eval_config[name] = seq[idx]
        elif isinstance(dist.lower, float):
            val = trial.suggest_float(name=name, low=dist.lower, high=dist.upper, log=dist.log)
            eval_config[name] = val
        else:
            val = trial.suggest_int(name=name, low=dist.lower, high=dist.upper, log=dist.log)
            eval_config[name] = val

    results = bench(eval_config)
    constraints = tuple(
        results[name] - threshold if bench.directions[name] == "min"
        else threshold - results[name] for name, threshold in bench.constraints.items()
    )
    trial.set_user_attr(_CONSTRAINT_KEY, constraints)
    return results["loss"]


def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
    return trial.user_attrs[_CONSTRAINT_KEY]


def get_study_name(args, dataset_name: str, quantiles: dict[str, float]) -> str:
    prefix = "ctpe" if args.ctpe else "original"
    study_name = f"{prefix}-{args.gamma_type}-{args.bench}-{dataset_name}-"
    study_name += "-".join([f"{name}={q}" for name, q in quantiles.items()])
    study_name += f"-{args.seed:0>2}"
    return study_name


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, choices=list(range(20)))
    parser.add_argument("--bench", type=str, choices=["jahs", "hpobench", "hpolib"])
    parser.add_argument("--ctpe", type=str, choices=["True", "False"])
    parser.add_argument("--dataset_id", default=0, type=int)
    parser.add_argument("--gamma_type", type=str, default="linear", choices=["sqrt", "linear"])
    args = parser.parse_args()
    args.ctpe = eval(args.ctpe)

    bench_cls = {"hpobench": HPOBench, "hpolib": HPOLib, "jahs": JAHSBench201}[args.bench]
    avail_constraint_names = bench_cls.avail_constraint_names
    dataset_name = bench_cls.dataset_names[args.dataset_id]
    data_path = os.path.join(os.environ["HOME"], f"hpo_benchmarks/{args.bench}/")
    q_choices = [0.01, 0.1, 0.5, 0.9, None]
    for q1 in q_choices:
        for q2 in q_choices:
            quantiles = {}
            if q1 is not None:
                quantiles[avail_constraint_names[0]] = q1
            if q2 is not None:
                quantiles[avail_constraint_names[1]] = q2
            if len(quantiles) == 0:
                continue

            bench = bench_cls(
                data_path=data_path,
                dataset_name=dataset_name,
                seed=args.seed,
                quantiles=quantiles,
            )
            run_study(
                objective=lambda trial: objective(trial, bench),
                constraints_func=constraints,
                seed=args.seed,
                ctpe=args.ctpe,
                gamma_type=args.gamma_type,
                study_name=get_study_name(args, dataset_name, quantiles),
                storage="sqlite:///ctpe-experiments.db",
                directions=["minimize"],
            )
