from data_preprocess import load_data
from train import train_model
from sklearn.model_selection import train_test_split
import torch
from data_preprocess import reshape_data, get_data_labels, get_device
from model_scripts.utils import accuracy
from torch import nn

if __name__ == "__main__":

    data = load_data('../../data/fashion-mnist_train.csv')
    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model, val_acc, train_accuracies = train_model(X_train, y_train, X_val, y_val, num_epochs=100, batch_size=256)

    # Save the trained model
    torch.save(model.state_dict(), "lenet5_final.pt")
    print(f"Training complete. Validation accuracy: {val_acc*100:.2f}%")


    #Run it on test data:
    model.eval()  # Set the model to evaluation mode

    data = load_data('../../data/fashion-mnist_test.csv')
    X = data.drop('label', axis=1)
    y = data['label']
    loss = nn.CrossEntropyLoss() 

    with torch.no_grad():  # Disable gradient calculation for validation
        X_test_tensor = reshape_data(X).to(get_device())
        y_test_tensor = get_data_labels(y).to(get_device())

        # Forward pass
        test_result = model(X_test_tensor)
        test_loss = loss(test_result, y_test_tensor)
        val_acc = accuracy(test_result, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {val_acc * 100:.2f}%')