import torch
from model_scripts.cnn_model import LeNet5
from data_preprocess import reshape_data, get_data_labels, get_device
from torch import nn, optim
from model_scripts.utils import accuracy
import numpy as np

def train_model(X_train, y_train, X_val, y_val, num_epochs=20, batch_size=128, learning_rate=0.001):

    model = LeNet5()
    model.to(get_device())
    loss = nn.CrossEntropyLoss() # Loss function for multi-class classification (using Softmax by default in PyTorch)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer with learning rate of 0.001

    # Training loop for batches of training data
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_accuracy = 0, 0

        for i in range(0, len(X_train), batch_size):
            X_batch = reshape_data(X_train[i:i+batch_size])
            y_batch = get_data_labels(y_train[i:i+batch_size])

            X_batch, y_batch = X_batch.to(get_device()), y_batch.to(get_device()) # Set to the same device as model

            # Forward pass
            train_result = model(X_batch)
            batch_loss = loss(train_result, y_batch) # Compute batch loss
            train_loss += batch_loss.item() * X_batch.size(0) # Accumulate loss over the epoch

            optimizer.zero_grad()           # Clear gradients from previous step

            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()

            acc = accuracy(train_result, y_batch).item() # Compute batch accuracy
            train_accuracy += acc * X_batch.size(0) # Accumulate accuracy over the epoch

        train_loss /= len(X_train)
        train_accuracy /= len(X_train)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')


    # Validation loop for whole validation set
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation for validation
        X_val_tensor = reshape_data(X_val).to(get_device())
        y_val_tensor = get_data_labels(y_val).to(get_device())

        # Forward pass
        val_result = model(X_val_tensor)
        val_loss = loss(val_result, y_val_tensor)
        val_acc = accuracy(val_result, y_val_tensor)
        print(f'Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_acc * 100:.2f}%')

    return model, val_acc.item(), np.array([train_accuracy])