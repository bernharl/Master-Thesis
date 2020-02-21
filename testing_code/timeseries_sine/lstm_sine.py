import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

device="cuda:1"

class SineData(torch.utils.data.Dataset):
    def __init__(self):
        tmp = torch.load("train_dataset.pt")
        self.y = torch.from_numpy(tmp[1].reshape(1, -1, 1)).float().to(device)
        self.x = torch.from_numpy(tmp[0].reshape(1, -1, 1)).float().to(device)
        self.num_samples = self.y.shape[0]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, i):
        return self.x[i], self.y[i] 

class Model(nn.Module):
    def __init__(self, hidden=10, layers=1):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        #print(x.shape)
        out = []
        for i in range(x.size(1)):
            output, (h_n, c_n) = self.lstm(x[:, i, :])
            print(output.shape)
            exit()
            out += [output]
        #print(output.shape, self.output(h_n).shape, self.output(output).shape)
        #exit()
        #print(h_n.shape)
        #out = self.output(output)
        #print(out)
        #print(self.output(output).shape)
        #exit()
        out = torch.stack(out, 1).squeeze(2)
        print(out.shape)
        exit()
        return out

data = SineData()
model = Model(hidden=51, layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
loss_func = nn.MSELoss()
model.train()
#print(data.x.shape)
#exit()
for i in range(1500):
    optimizer.zero_grad()
    #print(x)
    #exit()
    y_pred = model.forward(data.x)
    loss = loss_func(data.y, y_pred)
    print(f"Epoch {i}: {loss.item()}")
    loss.backward()
    optimizer.step()
    
with torch.no_grad():
    y_pred = model(data.x).detach().flatten().cpu()
plt.plot(data.x.detach().flatten().cpu(), data.y.detach().flatten().cpu(), label="true")
plt.plot(data.x.detach().flatten().cpu(), y_pred, label="pred")
plt.legend()
plt.savefig("split.png")
plt.close()
    