from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

class CamelsCl:
    def __init__(self, camels_root: Union[Path, str]):
        if isinstance(camels_root, str):
            camels_root = Path(camels_root)
        elif not isinstance(camels_root, Path):
            raise TypeError(f"camels_root must be PosixPath or string, not {type(camels_root)}")
        precip = camels_root / "4_CAMELScl_precip_cr2met"
        print(list(precip.rglob("*")))


if __name__ == "__main__":
    test = CamelsCl("/home/bernhard/git/datasets_masters/camels_ch")
