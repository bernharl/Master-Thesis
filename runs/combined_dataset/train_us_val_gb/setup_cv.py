from camelsml import split_basins, cross_validation_split, load_config


cfg = load_config(cfg_file="../camels_root_info.txt")
cross_validation_split(
    camels_root=cfg["camels_root"],
    basin_list="/home/bernhard/git/Master-Thesis/basin_lists/basin_list_us_specified.txt",
    k=5,
    test_split=0.25,
    store_folder="/home/bernhard/git/Master-Thesis/runs/combined_dataset/train_us_val_gb/cv",
    seed=19970204,
    dataset=cfg["dataset"],
    timeseries=cfg["timeseries"],
)

split_basins(
    camels_root=cfg["camels_root"],
    basin_list="/home/bernhard/git/Master-Thesis/basin_lists/basin_list_gb_specified.txt",
    split=[0, 0.65, 0.25],
    dataset=cfg["dataset"],
    timeseries=cfg["timeseries"],
    seed=19970204,
    normalize=False,
    store_folder="/home/bernhard/git/Master-Thesis/runs/combined_dataset/train_us_val_gb/gb_split",
)
