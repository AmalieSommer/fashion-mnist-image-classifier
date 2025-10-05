from data_preprocess import load_data
from train import train_model
from sklearn.model_selection import train_test_split
import torch

if __name__ == "__main__":

    data = load_data('data/fashion-mnist_train.csv')
    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model, val_acc, train_accuracies = train_model(X_train, y_train, X_val, y_val, num_epochs=50, batch_size=256)

    # Save the trained model
    torch.save(model.state_dict(), "lenet5_final.pt")
    print(f"Training complete. Validation accuracy: {val_acc*100:.2f}%")