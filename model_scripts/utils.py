import torch
from torchmetrics import Accuracy
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os


def accuracy(y_pred, y_true):
    return Accuracy(task="multiclass", num_classes=10).to('cuda' if torch.cuda.is_available() else 'cpu')(y_pred, y_true)


def processImg(img_path):
    process = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    img = Image.open(img_path).convert('L')
    img_tensor = process(img).unsqueeze(0)
    return img_tensor


def imgToNumpy(img_path, img_rows=28, img_cols=28):
    image = Image.open(img_path).convert('L')
    resized_image = image.resize((img_rows, img_cols))

    array = np.array(resized_image)
    return array.reshape(1, -1)


def reduceScaleFeatures(X, model):
    scaled_x = model.scaler.fit_transform(X)
    reduced_x = model.pca.fit_transform(scaled_x)
    return reduced_x


def reshapeTensor(X, img_row=28, img_col=28):
    return torch.tensor(X.to_numpy().reshape(-1, 1, img_row, img_col).astype('float32') / 255.0)


def getDataLabels(y):
    return torch.tensor(y.values.astype('int64'))


def getDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loadData(filepath):
    return pd.read_csv(filepath)


def getModelPath():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'saved_models')