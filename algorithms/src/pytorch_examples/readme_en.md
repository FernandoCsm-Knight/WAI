# Neural Networks with PyTorch - Examples and Best Practices

This repository contains a collection of neural network examples implemented using the PyTorch framework. The examples range from simple models to more complex architectures, providing a great opportunity to learn and practice neural network development.

## Attention: Data Sampling

It's important to note that many of the examples in this repository do not perform proper data sampling as they are just examples of using PyTorch. Instead, they use the same training data for testing, which can lead to misleading results and overestimation of the model's performance.
To obtain an accurate evaluation of a model's performance, it is crucial to properly separate the data into training, validation, and test sets. Using the same training data for testing can lead to overfitting, where the model excessively adjusts to the training data and does not generalize well to unseen data.

### Correct Sampling Methods

Here are some correct sampling methods that can be used instead of testing with the training data itself:

- **Holdout:** In this method, the data is divided into a fixed percentage for training (usually 70-80%) and the rest for testing. This ensures that the model is evaluated on data not seen during training.

​```python
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
​```

- **Cross-Validation:** Cross-validation involves dividing the data into k equal-sized subsets (folds). The model is trained on k-1 folds and tested on the remaining fold. This process is repeated k times, using each fold once as the test set. The average of the results from each iteration provides a more robust estimate of the model's performance.

​```python
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import torch

kf = KFold(n_splits=5, shuffle=True, random_state=42)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
​```

- **Random Sampling:** In this method, the data is randomly divided into training and test sets. This helps avoid bias in data selection and provides a more realistic evaluation of the model's performance.

​```python
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
​```

- **Stratified Sampling:** Similar to random sampling, but ensures that the class distribution is preserved in the training and test sets. This is especially useful when dealing with imbalanced datasets.

​```python
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
import torch

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
​```

> By using these sampling methods, we can obtain a more accurate and reliable evaluation of the performance of our neural networks.
>
> **Examples of Neural Networks:**
>
> - Simple Linear Regression
>
> - Logistic Regression
>
> - Feedforward Neural Networks (MLP)
>
> - Convolutional Neural Networks (CNN)
>
> - Recurrent Neural Networks (RNN)
>
> - Autoencoders
>
> - Generative Adversarial Networks (GANs)