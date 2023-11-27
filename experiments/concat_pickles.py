import pickle


results = {}
for bench in ["hpolib", "hpobench", "jahs"]:
    results.update(pickle.load(open(f"ctpe-experiments-{bench}.pkl", mode="rb")))

with open("ctpe-experiments.pkl", mode="wb") as f:
    pickle.dump(results, f)
