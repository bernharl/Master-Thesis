import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class Model(nn.Module):
    def __init__(self):
        hidden = 200
        layers = 2
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        #print(output.shape, self.output(h_n).shape, self.output(output).shape)
        #exit()
        return self.output(output)


data = torch.load("train_dataset.pt")
y_true = torch.from_numpy(data[1].reshape(-1, 1, 1)).float()
x = torch.from_numpy(data[0].reshape(-1, 1, 1)).float()

model = Model()
criterion = nn.MSELoss()
optimizer = optim.LBFGS(model.parameters(), lr=0.05)
for i in range(15):
    print(f"Epoch {i}")

    def closure():
        optimizer.zero_grad()
        y = model(x)
        loss = criterion(y, y_true)
        print(f"Loss: {loss}")
        loss.backward()
        return loss

    optimizer.step(closure)

test = torch.load("test_dataset.pt")
combined = np.append(data, test, axis=1)
#print(combined.shape, test.shape)
y_true_test = torch.from_numpy(test[1].reshape(-1, 1, 1)).float()
x_test = torch.from_numpy(test[0].reshape(-1, 1, 1)).float()

y_true_combined = torch.from_numpy(combined[1].reshape(-1, 1, 1)).float()
x_combined = torch.from_numpy(combined[0].reshape(-1, 1, 1)).float()

with torch.no_grad():
    y = model(x).flatten()
    y_test = model(x_test).flatten()
    y_combined = model(x_combined).flatten()
x = x.flatten()
x_test = x_test.flatten()
x_combined = x_combined.flatten()
y_true = y_true.flatten()
y_true_test = y_true_test.flatten()
y_true_combined = y_true_combined.flatten()
plt.plot(x, y, label="pred")
plt.plot(x, y_true, label="true")
plt.plot(x_test, y_test, label="pred test")
plt.plot(x_test, y_true_test, label="true test")
plt.legend()
plt.show()
plt.plot(x_combined, y_combined, label="pred")
plt.plot(x_combined, y_true_combined, label="true")
plt.legend()
plt.show()
