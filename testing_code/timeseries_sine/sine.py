import numpy as np
import torch

sample = 100
#test_sample = 50
f = 8

train_data = np.arange(0, sample)
test_data = np.arange(sample, 2*sample)# + test_sample)

#train_data = np.linspace(0, L_train, int(L_train * f * multiplier), endpoint=True)
train_data = np.array([train_data, np.sin(2 * np.pi * train_data * f / sample)])
torch.save(train_data, "train_dataset.pt")

#test_data = np.linspace(
#    L_test, L_train, int(f * (L_test - L_train) * multiplier), endpoint=True
#)
test_data = np.array([test_data, np.sin(2 * np.pi * test_data * f / sample)])
torch.save(test_data, "test_dataset.pt")
