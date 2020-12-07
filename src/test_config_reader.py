from pathlib import Path

from camelsml import load_config, train

cfg = load_config("training_runs/test/test.txt", device="cuda:1", num_workers=8)
cfg["split_train_test_folder"] = Path(
    "/home/bernhard/git/ealstm_regional_modeling_camels_gb/data/split/single_split_0911_1010"
)
print(cfg)
train(cfg)
