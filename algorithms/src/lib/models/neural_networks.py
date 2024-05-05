import numpy as np
import matplotlib.pyplot as plt

from algorithms.src.lib.settings import Paths

class Perceptron:
    def __init__(self):
        self.weights = None
        self.bias = None
        
    def fit(self, data, target, epochs=1000, alpha=0.1):
        if type(data) != np.ndarray:
            data = np.array(data)
            
        if type(target) != np.ndarray:
            target = np.array(target)
        
        self.weights = np.random.randn(data.shape[1])
        self.bias = np.random.randn()
        
        for _ in range(epochs):
            for xi, yi in zip(data, target):
                update = alpha * (yi - self.predict(xi))
                self.weights += update * xi
                self.bias += update
                
    def predict(self, data):
        return np.where(np.dot(data, self.weights) + self.bias > 0, 1, 0)

    def plot(self, data, target):
        if type(data) != np.ndarray:
            data = np.array(data)
            
        if type(target) != np.ndarray:
            target = np.array(target)
            
        if data.shape[1] != 2:
            raise ValueError("Data must have 2 features")
        
        x = np.linspace(data[:, 0].min() - 1, data[:, 0].max() + 1, 1000)
        y = - (self.weights[0] * x + self.bias) / self.weights[1]
        
        plt.scatter(data[:, 0], data[:, 1], c=target)
        plt.plot(x, y)
        plt.savefig(Paths.IMG_PATH + "perceptron.png")
    
    def describe(self, data, target):
        print(f"Accuracy: {self.accuracy(data, target)}")
        print(f"F1 Score: {self.f1_score(data, target)}")
        print(f"Recall: {self.recall(data, target)}")
        print(f"Precision: {self.precision(data, target)}")
    
    def __str__(self):
        return f"Perceptron(weights={self.weights}, bias={self.bias})"
    
    def __repr__(self):
        return str(self)
        