from pathlib import Path

import numpy as np

from camelsml.utils import get_basin_list
from camelsml import split_basins, load_config
from camelsml.split_basins import create_normalization_file


basins_test = get_basin_list("train_us_val_gb/cv/cross_validation_seed_19970204/basins_test.txt")
basins_val = get_basin_list("train_us_val_gb/cv/cross_validation_seed_19970204/0/basins_val.txt")
basins_train = get_basin_list("train_us_val_gb/cv/cross_validation_seed_19970204/0/basins_train.txt")

basins_train_save = []
for basin in basins_val:
    basins_train_save.append(basin)
for basin in basins_train:
    basins_train_save.append(basin)

refit_dir = Path("train_us_val_gb") / "refit_splits"
refit_dir.mkdir(exist_ok=True)
np.savetxt(refit_dir / "basins_test_us.txt", basins_test, fmt="%s")
np.savetxt(refit_dir / "basins_train_us.txt", basins_train, fmt="%s")



cfg = load_config(cfg_file="camels_root_info.txt")
create_normalization_file(camels_root=cfg["camels_root"], train_basin_list = refit_dir / "basins_train_us.txt", dataset=cfg["dataset"], timeseries=cfg["timeseries"])
