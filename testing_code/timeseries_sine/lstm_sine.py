import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

device="cuda:1"

class SineData(torch.utils.data.Dataset):
    def __init__(self):
        tmp = torch.load("train_dataset.pt")
        self.y = torch.from_numpy(tmp[1][:, :, np.newaxis]).float().to(device)
        self.x = torch.from_numpy(tmp[0][:, :, np.newaxis]).float().to(device)
        
        tmp = torch.load("test_dataset.pt")
        self.y_test = torch.from_numpy(tmp[1][:, :, np.newaxis]).float().to(device)
        self.x_test = torch.from_numpy(tmp[0][:, :, np.newaxis]).float().to(device)
        #print(self.y.shape)
        #exit()
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
        #out = []
        """for i in range(x.size(1)):
            output, (h_n, c_n) = self.lstm(x[:, i, :])
            #print(output.shape)
            exit()
            out += [output]"""
        output, (h_n, c_n) = self.lstm(x)
        #print(output.shape, h_n.shape, c_n.shape)
        #exit()
        #print(output.shape, self.output(h_n).shape, self.output(output).shape)
        #exit()
        #print(h_n.shape)
        #out = self.output(output)
        #print(out)
        #print(self.output(output).shape)
        #exit()
        #out = torch.stack(out, 1).squeeze(2)
        #print(out.shape)
        #exit()
        return self.output(output)

data = SineData()
model = Model(51, 2).to(device)
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)
loss_func = nn.MSELoss()
model.train()
#print(data.y.shape)
#exit()
for i in range(50):
    def closure():
        optimizer.zero_grad()
        #print(x)
        #exit()
        y_pred = model.forward(data.x)
        #print(y_pred.shape)
        #exit()
        loss = loss_func(data.y, y_pred)
        print(f"Epoch {i}: {loss.item()}")
        loss.backward()
        return loss
    optimizer.step(closure)
    
with torch.no_grad():
    y_pred = model(data.x).detach()[0].flatten().cpu()
    y_pred_test = model(data.x_test).detach()[0].flatten().cpu()
plt.plot(data.x.detach()[0].flatten().cpu(), data.y.detach()[0].flatten().cpu(), label="true")
plt.plot(data.x.detach()[0].flatten().cpu(), y_pred, label="pred")

plt.plot(data.x_test.detach()[0].flatten().cpu(), data.y_test.detach()[0].flatten().cpu(), label="true test")
plt.plot(data.x_test.detach()[0].flatten().cpu(), y_pred_test, label="pred test")

plt.legend()
plt.savefig("split.png")
plt.close()
    