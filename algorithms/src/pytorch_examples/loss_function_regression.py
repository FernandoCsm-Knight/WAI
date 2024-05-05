import torch
from torch import nn 
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size*2)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_size*2, hidden_size*2)
        self.tang = nn.Tanh()
        self.hidden3 = nn.Linear(hidden_size*2, hidden_size)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        return self.output(self.relu2(self.hidden3(self.tang(self.hidden2(self.relu(self.hidden(x)))))))
    
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
db = load_diabetes()

scaler = MinMaxScaler()
data = scaler.fit_transform(db.data)

data = torch.Tensor(db.data).to(device)
target = torch.Tensor(db.target).to(device)
    
net = Net(data.shape[1], 32, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for _ in range(10000):
    optimizer.zero_grad()
    output = net(data)
    loss = criterion(output.squeeze(), target)
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item()}")
    if loss.item() < 0.1:
        break
    

pred = net(data).squeeze().cpu().detach().numpy()

from sklearn.metrics import mean_squared_error

print(f"MSE: {mean_squared_error(target.cpu().numpy(), pred)}")