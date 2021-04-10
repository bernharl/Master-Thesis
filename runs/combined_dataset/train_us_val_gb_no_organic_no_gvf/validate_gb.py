from time import sleep
from pathlib import Path

from camelsml import evaluate, load_config


cfg = load_config("run_config.txt", device="cuda:0")
cfg["val_basin_file"] = Path(
    f"gb_split/split_seed_19970204/basins_validation.txt"
)
for k in range(3,4):
    runs = list(Path(f"{k}").glob("*"))
    print(runs)
    if len(runs) != 1:
        raise RuntimeError(f"Amount of runs per cross val should be 1, not {len(runs)}")
    cfg["run_dir"] = runs[0]
    cfg["train_basin_file"] = Path(
        f"cv/cross_validation_seed_19970204/{k}/basins_train.txt"
    )
    cfg["eval_dir"] = Path(f"val_gb/{k}")
    if k == 0:
        start = 25
    else:
        start = 1
    for i in range(start, cfg["epochs"] + 1):
        evaluate(cfg, split="val", epoch=i)
