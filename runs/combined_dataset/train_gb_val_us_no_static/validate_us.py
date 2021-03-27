from pathlib import Path

from camelsml import evaluate, load_config

cfg = load_config("run_config.txt")
cfg["val_basin_file"] = Path(
    f"../train_gb_val_us/us_split/split_seed_19970204/basins_validation.txt"
)
for k in range(5):
    runs = list(Path(f"{k}").glob("*"))
    print(runs)
    if len(runs) != 1:
        raise RuntimeError(f"Amount of runs per cross val should be 1, not {len(runs)}")
    cfg["run_dir"] = runs[0]
    cfg["train_basin_file"] = Path(
        f"../train_gb_val_us/cv/cross_validation_seed_19970204/{k}/basins_train.txt"
    )
    cfg["eval_dir"] = Path(f"val_us/{k}")
    for i in range(1, cfg["epochs"] + 1):
        evaluate(cfg, split="val", epoch=i)
