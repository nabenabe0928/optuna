from __future__ import annotations

import pickle

import numpy as np

import optuna


STORAGE = "sqlite:///ctpe-experiments.db"


def extract_trials_from_study(study_name: str) -> dict[str, np.ndarray]:
    study = optuna.load_study(study_name=study_name, storage=STORAGE)
    results = {
        "loss": np.asarray([t.value for t in study.trials]),
        "feasible": np.asarray([all(c <= 0 for c in t.user_attrs["constraints"]) for t in study.trials]),
    }
    return results


if __name__ == "__main__":
    study_names = optuna.get_all_study_names(STORAGE)
    results = {}
    for i, study_name in enumerate(study_names):
        if (i + 1) % 50 == 0:
            print(i + 1)

        results[study_name] = extract_trials_from_study(study_name)

    with open("ctpe-experiments.pkl", mode="wb") as f:
        pickle.dump(results, f)
