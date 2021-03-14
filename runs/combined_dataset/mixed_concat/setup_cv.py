from pathlib import Path

import numpy as np

from camelsml import (
    split_basins,
    cross_validation_split,
    load_config,
    combine_cv_datasets,
)


cfg = load_config(cfg_file="../camels_root_info.txt")
cv_folder_us = Path(
    "/home/bernhard/git/Master-Thesis/runs/combined_dataset/train_us_val_gb/cv"
)
cv_folder_gb = Path(
    "/home/bernhard/git/Master-Thesis/runs/combined_dataset/train_gb_val_us/cv"
)
store_folder = Path("/home/bernhard/git/Master-Thesis/runs/combined_dataset/mixed/cv")


combine_cv_datasets(
    cv_folder_1=cv_folder_us,
    cv_folder_2=cv_folder_gb,
    k=5,
    seed=19970204,
    normalize=True,
    store_folder=store_folder,
    dataset=cfg["dataset"],
    timeseries=cfg["timeseries"],
    camels_root=cfg["camels_root"],
)
