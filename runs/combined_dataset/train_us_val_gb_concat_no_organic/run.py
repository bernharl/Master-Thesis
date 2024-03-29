from pathlib import Path

from camelsml import load_config, train

cfg = load_config(cfg_file="run_config.txt", device="cuda:1", num_workers=32)
cfg["test_basin_file"] = Path(
   "/home/bernharl/git/Master-Thesis/runs/combined_dataset/train_us_val_gb/cv/cross_validation_seed_19970204/basins_test.txt" 
)
for i in range(2,5):
    cfg["run_dir"] = Path(
        f"/work/bernharl/train_us_val_gb_concat_no_organic/{i}"
    )
    cfg["train_basin_file"] = Path(
        f"/home/bernharl/git/Master-Thesis/runs/combined_dataset/train_us_val_gb/cv/cross_validation_seed_19970204/{i}/basins_train.txt"
    )
    cfg["val_basin_file"] = Path(
            f"/home/bernharl/git/Master-Thesis/runs/combined_dataset/train_us_val_gb/cv/cross_validation_seed_19970204/{i}/basins_val.txt"
    )
    train(cfg)
