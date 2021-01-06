from camelsml import cross_validation_split, load_config

cfg = load_config(cfg_file="camels_root_info.txt")
cross_validation_split(
    camels_root=cfg["camels_root"],
    basin_list="/home/bernhard/git/ealstm_regional_modeling_camels_gb/data/basin_list.txt",
    k=5,
    test_split=0.25,
    store_folder="/home/bernhard/git/Master-Thesis/runs/correlation_reduction/cross_validation",
    seed=19970204,
)
