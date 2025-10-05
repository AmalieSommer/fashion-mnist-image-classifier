import torch
from torchmetrics import Accuracy

def accuracy(y_pred, y_true):
    return Accuracy(task="multiclass", num_classes=10).to('cuda' if torch.cuda.is_available() else 'cpu')(y_pred, y_true)