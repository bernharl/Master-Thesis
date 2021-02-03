from pathlib import Path

from camelsml import evaluate, load_config

"""
This exists in case of me being dumb
"""
cfg = load_config("run_config.txt")
for k in range(5):
    runs = list(Path("0").glob("*"))
    if len(runs) != 1:
        raise RuntimeError(f"Amount of runs per cross val should be 1, not {len(runs)}")
    cfg["run_dir"] = runs[0]
    cfg["train_basin_file"] = Path(
        f"/home/bernhard/git/Master-Thesis/runs/camels_us/cross_validation/cross_validation_seed_19970204/{k}/basins_train.txt"
    )
    cfg["val_basin_file"] = Path(
        f"/home/bernhard/git/Master-Thesis/runs/camels_us/cross_validation/cross_validation_seed_19970204/{k}/basins_val.txt"
    )
    for i in range(1, cfg["epochs"] + 1):
        evaluate(cfg, split="val", epoch=i)
