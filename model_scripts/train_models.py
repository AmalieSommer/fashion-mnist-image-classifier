import torch
from cnn_model import LeNet5
from logReg_model import CustomLogisticRegression
from randForest_model import CustomRandomForest
from torch import nn, optim
from utils import accuracy, reshapeTensor, getDataLabels, getDevice, reduceScaleFeatures, loadData, getModelPath
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys


def trainLogReg(X_train, y_train, X_val, y_val):
    
    model = CustomLogisticRegression()
    #X_train_transformed = reduceScaleFeatures(X_train)
    #X_val_transformed = reduceScaleFeatures(X_val)

    #Fit model onto training set:
    model.fit(X_train, y_train)
    #X_val_transformed = model.pca.transform(model.scaler.transform(X_val))

    #Evaluate model performance:
    y_pred = model.predictClass(X_val)
    score = accuracy_score(y_val, y_pred)
    print(f'Logistic Regression Model Accuracy on validation data: {score}, and in percentage: {score * 100:.2f}%')

    model.saveModel() #Save the fitted model.



def trainRandForest(X_train, y_train, X_val, y_val):
    model = CustomRandomForest()
    X_train_transformed = reduceScaleFeatures(X_train)
    X_val_transformed = reduceScaleFeatures(X_val)

    model.fit(X_train_transformed, y_train)

    y_pred = model.predictClass(X_val_transformed)
    score = accuracy_score(y_val, y_pred)
    print(f'Random Forest Model Accuracy on test data: {score}, and in percentage: {score * 100:.2f}%')

    model.saveModel() 


def trainCnnModel(X_train, y_train, X_val, y_val, num_epochs=20, batch_size=128, learning_rate=0.001):
    
    model = LeNet5()
    model.to(getDevice())
    loss = nn.CrossEntropyLoss() # Loss function for multi-class classification (using Softmax by default in PyTorch)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer with learning rate of 0.001

    # Training loop for batches of training data
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_accuracy = 0, 0

        for i in range(0, len(X_train), batch_size):
            X_batch = reshapeTensor(X_train[i:i+batch_size])
            y_batch = getDataLabels(y_train[i:i+batch_size])

            X_batch, y_batch = X_batch.to(getDevice()), y_batch.to(getDevice()) # Set to the same device as model

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
        X_val_tensor = reshapeTensor(X_val).to(getDevice())
        y_val_tensor = getDataLabels(y_val).to(getDevice())

        # Forward pass
        val_result = model(X_val_tensor)
        val_loss = loss(val_result, y_val_tensor)
        val_acc = accuracy(val_result, y_val_tensor)
        print(f'Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_acc * 100:.2f}%')

    model.saveModel() #Save the trained model.



if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Missing inpupt!')
        sys.exit(1) # Exits the script when error caught.

    selectedModel = sys.argv[1]

    # Get and prepare the data:
    data = loadData('data/fashion-mnist_train.csv')
    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    if selectedModel == 'cnn':
        trainCnnModel(X_train, y_train, X_val, y_val)
    
    elif selectedModel == 'lr':
        trainLogReg(X_train, y_train, X_val, y_val)

    elif selectedModel == 'rf':
        trainRandForest(X_train, y_train, X_val, y_val)

    else:
        print('Wrong input!')
        sys.exit(1) # Exits the script cause of wrong input.