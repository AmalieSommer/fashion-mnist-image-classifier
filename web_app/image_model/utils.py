import torch
from torchmetrics import Accuracy
import torchvision.transforms as transforms
from PIL import Image

def accuracy(y_pred, y_true):
    return Accuracy(task="multiclass", num_classes=10).to('cuda' if torch.cuda.is_available() else 'cpu')(y_pred, y_true)


def process_img(img_path):
    process = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    img = Image.open(img_path).convert('L')
    img_tensor = process(img).unsqueeze(0)
    return img_tensor