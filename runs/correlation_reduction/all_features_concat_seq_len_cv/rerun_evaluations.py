from pathlib import Path

from camelsml import evaluate, load_config

"""
This exists in case of me being dumb
"""
cfg = load_config("run_config.txt", device="cuda:0", num_workers=60)
for i in range(4,5):
    run = list(Path(f"{i}").glob("*"))
    if len(run) != 1:
        raise RuntimeError(f"Found {len(run)} runs, need 1.")
    cfg["run_dir"] = run[0]
    cfg["train_basin_file"] = Path(
        f"/home/bernhard/git/Master-Thesis/runs/correlation_reduction/cross_validation/cross_validation_seed_19970204/{i}/basins_train.txt"
    )
    cfg["val_basin_file"] = Path(
        f"/home/bernhard/git/Master-Thesis/runs/correlation_reduction/cross_validation/cross_validation_seed_19970204/{i}/basins_val.txt"
    )
    for i in range(1, cfg["epochs"] + 1):
        evaluate(cfg, split="val", epoch=i)
