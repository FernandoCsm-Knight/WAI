from sklearn import datasets
from torch import nn 
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

wine = datasets.load_wine()
data = torch.Tensor(wine.data).to(device)
target = torch.Tensor(wine.target).to(device)
labels = wine.target_names

class WiniClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WiniClassifier, self).__init__()
        
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        return self.output(self.relu(self.hidden(x)))

model = WiniClassifier(data.shape[1], 32, len(labels))
model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for _ in range(3000):
    optimizer.zero_grad()
    pred = model(data)
    loss = criterion(pred, target.long())
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item()}")
    
pred = model(data)

softmax = nn.Softmax(dim=1)
print(softmax(pred[:5]))

pred = pred.argmax(dim=1).cpu().numpy()

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

print(f"Accuracy: {accuracy_score(target.cpu().numpy(), pred)}")
print(f"F1 Score: {f1_score(target.cpu().numpy(), pred, average='weighted')}")
print(f"Recall: {recall_score(target.cpu().numpy(), pred, average='weighted')}")
print(f"Precision: {precision_score(target.cpu().numpy(), pred, average='weighted')}")

