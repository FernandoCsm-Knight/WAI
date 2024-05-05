import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class Paths:
    IMG_PATH = 'images/'
    DATA_PATH = 'data/'
    
    def ensure_paths_exists():
        if not os.path.exists(Paths.IMG_PATH):
            os.makedirs(Paths.IMG_PATH)
        
        if not os.path.exists(Paths.DATA_PATH):
            os.makedirs(Paths.DATA_PATH)

Paths.ensure_paths_exists()

args = {
    'batch_size': 32,
    'num_workers': 4,
}

if torch.cuda.is_available():
    args['device'] = torch.device("cuda")
else:
    args['device'] = torch.device("cpu")

download = not os.path.exists(f'{Paths.DATA_PATH}MNIST')

db_train = datasets.MNIST(Paths.DATA_PATH,
                    train=True,
                    transform=transforms.ToTensor(),
                    download=download)

db_test = datasets.MNIST(Paths.DATA_PATH,
                    train=False,
                    transform=transforms.ToTensor(),
                    download=False)

data, label = db_train[0]

plt.figure()
plt.imshow(data[0])
plt.savefig(f'{Paths.IMG_PATH}mnist_sample.png')

print(f'Train set {len(db_train)}', f'Test set {len(db_test)}', end='\n')

train_loader = DataLoader(db_train, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
test_loader = DataLoader(db_test, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.output(self.features(x))
    

input_size = 28*28 
hidden_size = 128
output_size = 10

net = Net(input_size, hidden_size, output_size).to(args['device'])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), weight_decay=0.0005, lr=0.01)

for epoch in range(10):
    epoch_loss = []
    for batch in train_loader:
        data, target = batch
        data, target = data.to(args['device']), target.to(args['device'])
        
        optimizer.zero_grad()
        pred = net(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        
        epoch_loss.append(loss.cpu().data)

    epoch_loss = np.asarray(epoch_loss)
    print(f'Epoch {epoch} - Loss: {epoch_loss.mean()} +- {epoch_loss.std()}')