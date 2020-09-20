from pathlib import Path
from itertools import combinations

import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import CamelsTXT
from train_utils import train_epoch, eval_model, calc_nse
from model import Model

CAMELS_ROOT = Path(
    "/home/bernhard/git/datasets_masters/camels_gb/catalogue.ceh.ac.uk/datastore/eidchub/8344e4f3-d2ea-44f5-8afa-86d2987543a9/"
)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
basin = (
    "46003"  # can be changed to any 8-digit basin id contained in the CAMELS data set
)

all_features = [
    "precipitation",
    "temperature",
    "peti",
    "humidity",
    "shortwave_rad",
    "longwave_rad",
    "windspeed",
]
all_feature_combos = []
for r in range(1, len(all_features) + 1):
    for subset in combinations(all_features, r):
        all_feature_combos.append(list(subset))
max_nse = -np.inf
for features in all_feature_combos:
    hidden_size = 10  # Number of LSTM cell
    input_size = len(features)
    dropout_rate = 0.0  # Dropout rate of the final fully connected Layer [0.0, 1.0]
    learning_rate = 1e-4  # Learning rate used to update the weights
    sequence_length = 365  # Length of the meteorological record provided to the network

    ##############
    # Data set up#
    ##############

    # Training data
    start_date = pd.to_datetime("1970-10-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("1995-09-30", format="%Y-%m-%d")
    ds_train = CamelsTXT(
        basin=basin,
        camels_root=CAMELS_ROOT,
        features=features,
        seq_length=sequence_length,
        period="train",
        dates=[start_date, end_date],
    )
    tr_loader = DataLoader(ds_train, batch_size=256, shuffle=True)

    # Validation data. We use the feature means/stds of the training period for normalization
    means = ds_train.get_means()
    stds = ds_train.get_stds()
    start_date = pd.to_datetime("1995-10-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("2000-09-30", format="%Y-%m-%d")
    ds_val = CamelsTXT(
        basin=basin,
        camels_root=CAMELS_ROOT,
        features=features,
        seq_length=sequence_length,
        period="eval",
        dates=[start_date, end_date],
        means=means,
        stds=stds,
    )
    val_loader = DataLoader(ds_val, batch_size=2048, shuffle=False)

    # Test data. We use the feature means/stds of the training period for normalization
    start_date = pd.to_datetime("2000-10-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("2015-09-30", format="%Y-%m-%d")
    ds_test = CamelsTXT(
        basin=basin,
        camels_root=CAMELS_ROOT,
        features=features,
        seq_length=sequence_length,
        period="eval",
        dates=[start_date, end_date],
        means=means,
        stds=stds,
    )
    test_loader = DataLoader(ds_test, batch_size=2048, shuffle=False)

    #########################
    # Model, Optimizer, Loss#
    #########################

    model = Model(
        hidden_size=hidden_size, input_size=input_size, dropout_rate=dropout_rate
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()

    n_epochs = 100  # Number of training epochs

    for i in range(n_epochs):
        train_epoch(model, optimizer, tr_loader, loss_func, i + 1, DEVICE)
        obs, preds = eval_model(model, val_loader, DEVICE)
        preds = ds_val.local_rescale(preds.cpu().numpy(), variable="output")
        nse = calc_nse(obs.cpu().numpy(), preds)
        tqdm.write(f"Validation NSE: {nse:.2f}")

    # Evaluate on test set
    obs, preds = eval_model(model, test_loader, DEVICE)
    preds = ds_val.local_rescale(preds.cpu().numpy(), variable="output")
    obs = obs.cpu().numpy()
    nse = calc_nse(obs, preds)
    if nse > max_nse:
        max_nse = nse
        best_features = features
    # Plot results
    start_date = ds_test.dates[0]
    end_date = ds_test.dates[1] + pd.DateOffset(days=1)
    date_range = pd.date_range(start_date, end_date)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(date_range, obs, label="observation")
    ax.plot(date_range, preds, label="prediction")
    ax.legend()
    ax.set_title(f"Basin {basin} - Test set NSE: {nse:.3f}")
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_xlabel("Date")
    _ = ax.set_ylabel("Discharge (mm/d)")
    plt.savefig(f"test_{features}.png")
print(f"Best NSE: {max_nse} achieved using {best_features}")
