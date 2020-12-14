from camelsml import load_config, train, evaluate

cfg = load_config(cfg_file="run_config.txt", device="cuda:0", num_workers=24)
evaluate(cfg, split="val")
train(cfg)
