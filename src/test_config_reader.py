from pathlib import Path

from camelsml import load_config, train, split_basins

cfg = load_config("training_runs/test/test.txt", device="cuda:0", num_workers=24)
"""split_basins(
    cfg["camels_root"],
    "/home/bernhard/git/ealstm_regional_modeling_camels_gb/data/basin_list.txt",
    split=[0.67, 0.33],
    store_folder="training_runs",
    seed=1010,
)"""
train(cfg)
