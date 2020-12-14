from camelsml import load_config, train

cfg = load_config(cfg_file="run_config.txt", device="cuda:0", num_workers=24)
train(cfg)
