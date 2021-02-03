from pathlib import Path

from camelsml import load_config, train

cfg = load_config(cfg_file="run_config.txt", device="cuda:0", num_workers=24)
cfg["test_basin_file"] = Path(
    "/home/bernhard/git/Master-Thesis/runs/camels_us/cross_validation/cross_validation_seed_19970204/basins_test.txt"
)
for i in range(5):
    cfg["run_dir"] = Path(
        f"/home/bernhard/git/Master-Thesis/runs/camels_us/chosen_features_cv_us/{i}"
    )
    cfg["train_basin_file"] = Path(
        f"/home/bernhard/git/Master-Thesis/runs/camels_us/cross_validation/cross_validation_seed_19970204/{i}/basins_train.txt"
    )
    cfg["val_basin_file"] = Path(
        f"/home/bernhard/git/Master-Thesis/runs/camels_us/cross_validation/cross_validation_seed_19970204/{i}/basins_val.txt"
    )
    train(cfg)
