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


def get_study_name(args, dataset_name: str, quantiles: dict[str, float], seed: int) -> str:
    prefix = "ctpe" if args.ctpe else "original"
    study_name = f"{prefix}-{args.gamma_type}-{args.bench}-{dataset_name}-"
    study_name += "-".join([f"{name}={q}" for name, q in quantiles.items()])
    study_name += f"-{seed:0>2}"
    return study_name


def get_quantile(args, avail_constraint_names: list[str]) -> dict[str, float]:
    quantiles = {}
    if args.q1 != 1.0:
        quantiles[avail_constraint_names[0]] = args.q1
    if args.q2 != 1.0:
        quantiles[avail_constraint_names[1]] = args.q2

    return quantiles


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--bench", type=str, choices=["jahs", "hpobench", "hpolib"])
    parser.add_argument("--ctpe", type=str, choices=["True", "False"])
    parser.add_argument("--q1", type=float, choices=[0.01, 0.1, 0.5, 0.9, 1.0])
    parser.add_argument("--q2", type=float, choices=[0.01, 0.1, 0.5, 0.9, 1.0])
    parser.add_argument("--dataset_id", default=0, type=int)
    parser.add_argument("--gamma_type", type=str, default="linear", choices=["sqrt", "linear"])
    args = parser.parse_args()
    args.ctpe = eval(args.ctpe)
    return args


def main() -> None:
    args = get_args()
    storage = "sqlite:///ctpe-experiments.db"
    bench_cls = {"hpobench": HPOBench, "hpolib": HPOLib, "jahs": JAHSBench201}[args.bench]
    dataset_name = bench_cls.dataset_names[args.dataset_id]

    quantiles = get_quantile(args, bench_cls.avail_constraint_names)
    try:
        bench = bench_cls(
            data_path=os.path.join(os.environ["HOME"], f"hpo_benchmarks/{args.bench}/"),
            dataset_name=dataset_name,
            seed=None,
            quantiles=quantiles,
        )
    except ValueError:
        print("No feasible solution exists, so skip it.")
        return

    if len(quantiles) == 0:
        return

    for seed in range(20):
        bench.reseed(seed)
        study_name = get_study_name(args, dataset_name, quantiles, seed)
        if study_name in optuna.get_all_study_names(storage):
            print(f"Found {study_name}, so skip it.")
            continue

        run_study(
            objective=lambda trial: objective(trial, bench),
            constraints_func=constraints,
            seed=seed,
            ctpe=args.ctpe,
            gamma_type=args.gamma_type,
            study_name=study_name,
            storage=storage,
            directions=["minimize"],
        )


if __name__ == "__main__":
    main()
