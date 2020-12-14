from camelsml import load_config, split_basins
from camelsml.datautils import normalize_features
from camelsml.utils import get_basin_list

cfg = load_config("training_runs/test/test.txt")
split_basins(cfg["camels_root"],  "/home/bernhard/git/ealstm_regional_modeling_camels_gb/data/basin_list.txt", split=[])
# normalize_features("jeff", "inputs", cfg["train_basin_file"].parent)
basin_list = get_basin_list(cfg["train_basin_file"])
