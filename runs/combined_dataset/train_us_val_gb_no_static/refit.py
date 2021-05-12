from pathlib import Path

from camelsml import load_config, evaluate, train
from camelsml.utils import get_basin_list

cfg = load_config(cfg_file="run_config.txt", device="cuda:0", num_workers=30)

cfg["test_basin_file"] = (
    Path("..")
    / "train_us_val_gb"
    / "gb_split"
    / "split_seed_19970204"
    / "basins_test.txt"
)


cfg["train_basin_file"] = Path("..") / "refit_splits" / "basins_train.txt"
cfg["evaluate_on_epoch"] = False

cfg["run_dir"] = Path() / "refit"

cfg["epochs"] = 9

# train(cfg)
finished_run = list(Path(cfg["run_dir"]).glob("*"))
# if len(finished_run) != 1:
#    raise RuntimeError(
#        f"Amount of runs in folder needs to be 1, not {len(finished_run)}"
#    )
# cfg["run_dir"] = finished_run[0]

evaluate(cfg, split="test", epoch=cfg["epochs"], save_dir=cfg["run_dir"] / "gb")
cfg["test_basin_file"] = (
    Path("..") / "train_us_val_gb" / "refit_splits" / "basins_test.txt"
)
