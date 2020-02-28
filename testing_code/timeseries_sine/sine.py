import numpy as np
import torch

sample = 169
# test_sample = 50
f = 8
waves = 20

data = np.arange(0, 2 * sample) + np.random.randint(-4 * sample // f, 4 * sample // f, waves).reshape(
    waves, 1
)
train_data = data[:, :sample]
test_data = data[:, sample : 2 * sample]  # + test_sample)

# train_data = np.linspace(0, L_train, int(L_train * f * multiplier), endpoint=True)
train_data = np.array([train_data, np.sin(2 * np.pi * train_data * f / sample)])
#print(train_data, train_data.shape)
torch.save(train_data, "train_dataset.pt")

# test_data = np.linspace(
#    L_test, L_train, int(f * (L_test - L_train) * multiplier), endpoint=True
# )
test_data = np.array([test_data, np.sin(2 * np.pi * test_data * f / sample)])
torch.save(test_data, "test_dataset.pt")

out_of_sample = np.arange(100 * sample, 102 * sample)
out_of_sample = np.array(
    [out_of_sample, np.sin(2 * np.pi * out_of_sample * f / sample)]
)
#print(train_data.shape, out_of_sample.shape)
torch.save(out_of_sample, "oos.pt")
