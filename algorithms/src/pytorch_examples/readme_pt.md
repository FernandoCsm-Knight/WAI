# Redes Neurais com PyTorch - Exemplos e Melhores Práticas

Este repositório contém uma coleção de exemplos de redes neurais implementadas usando o framework PyTorch. Os exemplos abrangem desde modelos simples até arquiteturas mais complexas, proporcionando uma ótima oportunidade para aprender e praticar o desenvolvimento de redes neurais.

## Atenção: Amostragem de Dados

É importante ressaltar que muitos dos exemplos neste repositório não realizam a amostragem correta dos dados por serem apenas exemplos de utilização do pytorch. Em vez disso, eles utilizam os mesmos dados de treinamento para teste, o que pode levar a resultados enganosos e superestimação do desempenho do modelo.
Para obter uma avaliação precisa do desempenho de um modelo, é crucial separar adequadamente os dados em conjuntos de treinamento, validação e teste. Usar os mesmos dados de treinamento para teste pode levar ao overfitting, onde o modelo se ajusta excessivamente aos dados de treinamento e não generaliza bem para dados não vistos.

### Métodos de Amostragem Corretos

Aqui estão alguns métodos de amostragem corretos que podem ser utilizados no lugar de testes com os próprios dados de treinamento:

- **Holdout:** Neste método, os dados são divididos em uma porcentagem fixa para treinamento (geralmente 70-80%) e o restante para teste. Isso garante que o modelo seja avaliado em dados não vistos durante o treinamento.

```python
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
```

- **Validação Cruzada (Cross-Validation):** A validação cruzada envolve dividir os dados em k subconjuntos (folds) de tamanho igual. O modelo é treinado em k-1 folds e testado no fold restante. Esse processo é repetido k vezes, usando cada fold uma vez como conjunto de teste. A média dos resultados de cada iteração fornece uma estimativa mais robusta do desempenho do modelo.

```python
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
```

- **Amostragem Aleatória:** Neste método, os dados são divididos aleatoriamente em conjuntos de treinamento e teste. Isso ajuda a evitar viés na seleção dos dados e fornece uma avaliação mais realista do desempenho do modelo.

```python
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
```

- **Amostragem Estratificada:** Semelhante à amostragem aleatória, mas garante que a distribuição das classes seja preservada nos conjuntos de treinamento e teste. Isso é especialmente útil quando lidamos com conjuntos de dados desequilibrados.

```python
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
```

> Ao utilizar esses métodos de amostragem, podemos obter uma avaliação mais precisa e confiável do desempenho de nossas redes neurais.
>
> **Exemplos de Redes Neurais:**
>
> - Regressão Linear Simples
>
> - Regressão Logística
>
> - Redes Neurais Feedforward (MLP)
>
> - Redes Neurais Convolucionais (CNN)
>
> - Redes Neurais Recorrentes (RNN)
>
> - Autoencoders
>
> - Redes Adversárias Generativas (GANs)

