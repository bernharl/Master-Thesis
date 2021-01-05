from camelsml import evaluate, load_config
"""
This exists in case of me being dumb
"""
cfg = load_config("run_config.txt")
cfg["run_dir"] = cfg["run_dir"] / "run_1412_0900_seed19970204"
for i in range(1, cfg["epochs"]+1):
    evaluate(cfg, split="val", epoch=i)
