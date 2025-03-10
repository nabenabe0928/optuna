import numpy as np

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from tqdm import tqdm


n_params = 50
n_trials = 100000
X = np.random.random((n_trials, n_params)) * 2 - 1
Y = np.sum(X**2, axis=-1)
storage = JournalStorage(JournalFileBackend("./journal_demo.log"))
study = optuna.create_study(
    storage=storage, sampler=optuna.samplers.RandomSampler(), study_name=f"large-study"
)

dists = {f"x{i}": optuna.distributions.FloatDistribution(-1, 1) for i in range(n_params)}
print("Create Trials...")
trials = [
    optuna.trial.create_trial(
        params={f"x{i}": v for i, v in enumerate(x)}, distributions=dists, value=float(y)
    ) for x, y in tqdm(zip(X, Y), total=n_trials)
]
print("Add Trials...")
for t in tqdm(trials):
    study.add_trial(t)
