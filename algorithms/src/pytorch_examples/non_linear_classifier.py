import torch
from torch import nn
from sklearn.datasets import load_iris

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        hidden = self.relu(self.hidden(x))
        output = self.output(hidden)
        return output

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
db = load_iris()
data = torch.Tensor(db.data)
target = torch.Tensor(db.target)
labels = db.target_names

net = Net(4, 16, 3)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

data = data.to(device)

for _ in range(1000):
    optimizer.zero_grad()
    output = net(data)
    loss = criterion(output, target.long())
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item()}")
    
output = net(data).argmax(dim=1)
output = output.cpu().numpy()

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

print(f"Accuracy: {accuracy_score(target, output)}")
print(f"F1 Score: {f1_score(target, output, average='weighted')}")
print(f"Recall: {recall_score(target, output, average='weighted')}")
print(f"Precision: {precision_score(target, output, average='weighted')}")




