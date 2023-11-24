import json
import os
import time

from chpobench import HPOLib

from experiments.bench import constraints
from experiments.bench import objective
from experiments.bench import run_study

import optuna


def main():
    bench = HPOLib(
        data_path=os.path.join(os.environ["HOME"], f"hpo_benchmarks/hpolib/"),
        dataset_name=HPOLib.dataset_names[0],
        seed=None,
        quantiles={"model_size": 0.1, "runtime": 0.1},
    )
    results = {"ctpe-linear": [], "ctpe-sqrt": [], "original-linear": []}
    N_SEEDS = 20
    N_TRIALS = 1000

    for setup_name in results:
        ctpe = "ctpe" in setup_name
        gamma_type = setup_name.split("-")[1]
        for seed in range(N_SEEDS):
            bench.reseed(seed)
            start = time.time()
            study = run_study(
                objective=lambda trial: objective(trial, bench),
                constraints_func=constraints,
                seed=seed,
                ctpe=ctpe,
                gamma_type=gamma_type,
                study_name=None,
                storage=None,
                directions=["minimize"],
                n_trials=N_TRIALS,
            )
            results[setup_name].append([
                t.datetime_complete.timestamp() - start for t in study.trials
            ])
    
    with open("ctpe-speed-bench.json", mode="w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
