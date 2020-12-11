

from camelsml import split_basins

split_basins("/home/bernhard/git/ealstm_regional_modeling_camels_gb/data/basin_list.txt", split=[0.65, 0.1, 0.25], store_folder="/home/bernhard/git/Master-Thesis/runs/correlation_reduction/basin_splits", seed=1337)
