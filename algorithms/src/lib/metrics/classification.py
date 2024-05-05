import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms.src.lib.settings import Paths

class ConfusionMatrix:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.confusion_matrix = self._compute_confusion_matrix()
        Paths.ensure_paths_exists()

    def _compute_confusion_matrix(self):
        unique_labels = np.unique(self.y_true)
        
        if len(unique_labels) < len(np.unique(self.y_pred)):
            unique_labels = np.unique(self.y_pred)
                
        confusion_matrix = np.zeros((len(unique_labels),  len(unique_labels)), dtype=int)
        for i in range(len(self.y_true)):
            confusion_matrix[self.y_true[i], self.y_pred[i]] += 1
            
        return confusion_matrix

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def get_accuracy(self):
        return np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)

    def get_precision(self):
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)

    def get_recall(self):
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
    
    def get_specificity(self):
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)

    def true_positive_rate(self):
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
    
    def false_positive_rate(self):
        return (np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix, axis=1)
    
    def true_negative_rate(self):
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
    
    def false_negative_rate(self):
        return (np.sum(self.confusion_matrix, axis=1) - np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix, axis=1)
    
    def plot_roc_curve(self):
        tpr = self.true_positive_rate()
        fpr = self.false_positive_rate()
        
        if len(np.unique(tpr)) == 1 or len(np.unique(fpr)) == 1:
            tpr = np.append(tpr, 0)
            fpr = np.append(fpr, 0)
            tpr = np.append(tpr, 1)
            fpr = np.append(fpr, 1)
    
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='blue', label='ROC Curve')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random')
        
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve", fontsize=14)
        plt.legend()
    
        plt.tight_layout()
        plt.savefig(Paths.IMG_PATH + "roc_curve.png")

    def plot(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.confusion_matrix, annot=True, cmap='crest', fmt='d')
        
        plt.xlabel("Predicted Labels", fontsize=12)
        plt.ylabel("True Labels", fontsize=12)
        plt.title("Confusion Matrix", fontsize=14)
        plt.tight_layout()
        plt.savefig(Paths.IMG_PATH + "confusion_matrix.png")

    def get_f1_score(self):
        precision = self.get_precision()
        recall = self.get_recall()
        return 2 * precision * recall / (precision + recall)

    def get_metrics(self):
        return {
            "accuracy": self.get_accuracy(),
            "precision": self.get_precision(),
            "recall": self.get_recall(),
            "specificity": self.get_specificity(),
            "f1_score": self.get_f1_score(),
        }