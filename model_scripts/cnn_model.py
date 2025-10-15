# Define the model architecture - LeNet-5
from torch import nn
from torch.nn import functional as F
import torch
from utils import getModelPath
import os

class LeNet5(nn.Module):
    def __init__(self):
        
        """ Initialize the LeNet-5 model architecture 
        
        Parameters:
        in_channels (int): Number of input channels (e.g., 1 for grayscale images, 3 for RGB images)
        num_classes (int): Number of output classes for classification
        """
        super().__init__()

        # Input is padded to keep LeNet-5 original architecture (28x28x1 --> 32x32x1).
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)   # 28x28x1 --> 28x28x6
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                                          # 28x28x6 --> 14x14x6
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)  # 14x14x6 --> 10x10x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                                          # 10x10x16 --> 5x5x16 
        self.fc1 = nn.Linear(in_features=5*5*16, out_features=120)
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout layer with a dropout probability of 0.5
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.dropout2 = nn.Dropout(p=0.3)  # Dropout layer with a dropout probability of 0.3
        self.out = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.reshape(x.shape[0], -1)   # Flatten the tensor

        x = F.relu(self.fc1(x))         # First fully connected layer with ReLu activation
        x = self.dropout1(x)              # Apply dropout for regularization
        x = F.relu(self.fc2(x))         # Second fully connected layer with ReLu activation
        x = self.dropout2(x)              # Apply dropout for regularization
        x = self.out(x)
        return x                         # Output layer (logits for each class)
    

    def saveModel(self, filepath='lenet5_model.pt'):
        saved_model_path = os.path.join(getModelPath(), filepath)
        torch.save(self.state_dict(), saved_model_path)


    def loadModel(self, device, filepath='lenet5_model.pt'):
        saved_model_path = os.path.join(getModelPath(), filepath)
        self.load_state_dict(torch.load(saved_model_path, map_location=device, weights_only=True))
        self.to(device)