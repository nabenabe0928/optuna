import time

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


print("Prepare Journal Backend...")
journal_backend = JournalFileBackend("./journal_demo.log")
time.sleep(2)

print("Prepare Journal Storage...")
storage = JournalStorage(journal_backend)
time.sleep(2)

print("Load Study...")
study = optuna.create_study(
    storage=storage,
    sampler=optuna.samplers.RandomSampler(),
    study_name="small-study",
    load_if_exists=True,
)
time.sleep(2)
