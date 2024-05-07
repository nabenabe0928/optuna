from __future__ import annotations

from argparse import ArgumentParser
import itertools

import numpy as np

import optuna

import pandas as pd


class PermutationShiftL1:
    def __init__(self, n_items: int):
        rng = np.random.RandomState(42)
        self._perms = np.asarray(list(map(list, itertools.permutations(range(n_items)))))
        self._indices = list(range(len(self._perms)))
        self._n_items = n_items
        self._opt = self._perms[rng.randint(len(self._perms))] + n_items / 2

    @staticmethod
    def _distance(perm1: np.ndarray, perm2: np.ndarray) -> float:
        return np.sum(np.abs(np.subtract(perm1, perm2)))

    def distance(self, index1: int, index2: int) -> float:
        return self._distance(self._perms[index1], self._perms[index2])

    def __call__(self, trial: optuna.Trial) -> float:
        perm = self._perms[trial.suggest_categorical("index", self._indices)]
        shift = trial.suggest_float("shift", -self._n_items, self._n_items)
        return self._distance(perm + shift, self._opt)


class EmbeddingCosine:
    def __init__(self, n_choices: int, length: int):
        rng = np.random.RandomState(42)
        raw_embs = rng.random((n_choices, length))
        self._embs = raw_embs / np.linalg.norm(raw_embs, axis=1)[:, None]
        self._indices = list(range(len(self._embs)))
        self._opt = self._embs[rng.randint(n_choices)]

    @staticmethod
    def _distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        return 1 - np.dot(emb1, emb2)

    def distance(self, index1: int, index2: int) -> float:
        return self._distance(self._embs[index1], self._embs[index2])

    def __call__(self, trial: optuna.Trial) -> float:
        emb = self._embs[trial.suggest_categorical("index", self._indices)]
        return self._distance(emb, self._opt)


class CustomParzenEstimator(optuna.samplers._tpe.parzen_estimator._ParzenEstimator):
    _modify: bool = True
    _log_base: int = 6

    def _calculate_categorical_distributions(
        self,
        observations: np.ndarray,
        param_name: str,
        search_space: optuna.distributions.CategoricalDistribution,
        parameters: optuna.samplers._tpe.parzen_estimator._ParzenEstimatorParameters,
    ) -> optuna.samplers._tpe.probability_distributions._BatchedDistributions:
        choices = search_space.choices
        n_choices = len(choices)
        n_kernels = len(observations) + parameters.consider_prior
        assert parameters.prior_weight is not None
        weights = np.full(shape=(n_kernels, n_choices), fill_value=parameters.prior_weight / n_kernels)
        observed_indices = observations.astype(int)
        used_indices, rev_indices = np.unique(observed_indices, return_inverse=True)
        dist_func = parameters.categorical_distance_func[param_name]
        dists = np.array([[dist_func(choices[i], c) for c in choices] for i in used_indices])
        
        coef = np.log(n_kernels / parameters.prior_weight)
        if self._modify:
            coef *= np.log(n_choices) / np.log(self._log_base)
 
        cat_weights = np.exp(-((dists / np.max(dists, axis=1)[:, np.newaxis]) ** 2) * coef)
        weights[: len(observed_indices)] = cat_weights[rev_indices]
        weights /= weights.sum(axis=1, keepdims=True)
        return optuna.samplers._tpe.probability_distributions._BatchedCategoricalDistributions(weights)


def run_experiment(
    func: EmbeddingCosine | PermutationShiftL1,
    opt_choice: str,
    seed: int,
    n_trials: int,
    log_base: int,
) -> list[float]:
    if opt_choice.startswith("metric-tpe"):
        sampler = optuna.samplers.TPESampler(
            seed=seed, multivariate=True, categorical_distance_func={"index": func.distance}
        )
        CustomParzenEstimator._modify = not opt_choice.endswith("no-modification")
        CustomParzenEstimator._log_base = log_base
        sampler._parzen_estimator_cls = CustomParzenEstimator
    elif opt_choice == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    elif opt_choice == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        assert f"Got an unknown opt: {opt_choice}"

    study = optuna.create_study(sampler=sampler)
    study.optimize(func, n_trials=n_trials)
    return [float(t.value) for t in study.trials]


def main(n_seeds: int, n_trials: int = 100) -> None:
    funcs = {
        "cos-500-8": EmbeddingCosine(n_choices=500, length=8),
        "cos-1000-16": EmbeddingCosine(n_choices=1000, length=16),
        "perm-6": PermutationShiftL1(n_items=6),
        "perm-7": PermutationShiftL1(n_items=7),
    }
    opts = ["metric-tpe", "metric-tpe-no-modification", "tpe", "random"]
    data = []
    for opt_name in opts:
        for func_name, func in funcs.items():
            for seed in range(n_seeds):
                values = run_experiment(func, opt_choice=opt_name, seed=seed, n_trials=n_trials, log_base=6)
                data.append({"opt_name": opt_name, "func_name": func_name, "seed": seed, "values": values})
                pd.DataFrame(data).to_json("results.json")


def ablation_study(n_seeds: int, n_trials: int = 100) -> None:
    funcs = {
        "cos-500-8": EmbeddingCosine(n_choices=500, length=8),
        "cos-1000-16": EmbeddingCosine(n_choices=1000, length=16),
        "perm-6": PermutationShiftL1(n_items=6),
        "perm-7": PermutationShiftL1(n_items=7),
    }
    data = []
    for log_base in range(2, 11):
        for func_name, func in funcs.items():
            for seed in range(n_seeds):
                values = run_experiment(func, opt_choice="metric-tpe", seed=seed, n_trials=n_trials, log_base=log_base)
                data.append({"opt_name": "metric-tpe", "func_name": func_name, "seed": seed, "values": values, "log_base": log_base})
                pd.DataFrame(data).to_json("ablation-study.json")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=["main", "ablation"], required=True)
    args = parser.parse_args()
    if args.mode == "main":
        main(n_seeds=10, n_trials=100)
    elif args.mode == "ablation":
        ablation_study(n_seeds=10, n_trials=100)
    else:
        assert False, f"Got an unknown {args.mode=}."
