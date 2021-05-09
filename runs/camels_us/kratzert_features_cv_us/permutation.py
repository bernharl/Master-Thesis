from pathlib import Path
import pickle
import random

import numpy as np

from camelsml import permutation_test, load_config


permutation_folder = Path("permutation")
permutation_folder.mkdir(exist_ok=True)
cfg = load_config("run_config.txt", device="cuda:0", num_workers=32)
np.random.seed(cfg["seed"])
random.seed(cfg["seed"])
for i in range(5):
    save_path = permutation_folder / f"{i}"
    save_path.mkdir(exist_ok=True)
    cv_dir = list((Path().absolute() / f"{i}").glob("*"))
    if len(cv_dir) != 1:
        raise RuntimeError(f"cv_dir must contain only one run")
    else:
        cv_dir = cv_dir[0]
    cfg["run_dir"] = cv_dir
    cfg["train_basin_file"] = Path(
        f"../cross_validation/cross_validation_seed_19970204/{i}/basins_train.txt"
    )
    cfg["val_basin_file"] = Path(
        f"../cross_validation/cross_validation_seed_19970204/{i}/basins_val.txt"
    )
    permutation_score = permutation_test(cfg, k=2, epoch=13)
    with open(save_path / "i_list.pickle", "wb") as outfile:
        pickle.dump(permutation_score , outfile)
