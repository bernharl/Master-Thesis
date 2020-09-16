from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from dataset import CamelsTXT
from model import Model

CAMELS_ROOT = Path(
    "/home/bernhard/git/datasets_masters/camels_gb/catalogue.ceh.ac.uk/datastore/eidchub/8344e4f3-d2ea-44f5-8afa-86d2987543a9/"
)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
basin = (
    "46003"  # can be changed to any 8-digit basin id contained in the CAMELS data set
)
hidden_size = 10  # Number of LSTM cells
dropout_rate = 0.0  # Dropout rate of the final fully connected Layer [0.0, 1.0]
learning_rate = 1e-4  # Learning rate used to update the weights
sequence_length = 365  # Length of the meteorological record provided to the network

##############
# Data set up#
##############

# Training data
start_date = pd.to_datetime("1980-10-01", format="%Y-%m-%d")
end_date = pd.to_datetime("1995-09-30", format="%Y-%m-%d")
ds_train = CamelsTXT(
    basin,
    CAMELS_ROOT,
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
    basin,
    CAMELS_ROOT,
    seq_length=sequence_length,
    period="eval",
    dates=[start_date, end_date],
    means=means,
    stds=stds,
)
val_loader = DataLoader(ds_val, batch_size=2048, shuffle=False)

# Test data. We use the feature means/stds of the training period for normalization
start_date = pd.to_datetime("2000-10-01", format="%Y-%m-%d")
end_date = pd.to_datetime("2010-09-30", format="%Y-%m-%d")
ds_test = CamelsTXT(
    basin,
    CAMELS_ROOT,
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

# Here we create our model, feel free
model = Model(hidden_size=hidden_size, dropout_rate=dropout_rate).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()
