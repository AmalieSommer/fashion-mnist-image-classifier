import torch
import pandas as pd


def reshape_data(X, img_rows=28, img_cols=28):
    """ Reshape the input data to have a single channel (grayscale images)
    
    Parameters:
    X (tensor): Input data to be reshaped
    img_rows (int): Number of rows in the image
    img_cols (int): Number of columns in the image
    
    Returns:
    tensor: Reshaped data with shape (num_samples, 1, img_rows, img_cols)
    """
    return torch.tensor(X.to_numpy().reshape(-1, 1, img_rows, img_cols).astype('float32') / 255.0)

def get_data_labels(y):
    """ Convert labels to tensor format
    
    Parameters:
    y (numpy array or pandas Series): Input labels
    
    Returns:
    tensor: Labels converted to tensor format
    """
    return torch.tensor(y.values.astype('int64'))

def get_device():
    """ Set the device to GPU if available, otherwise use CPU
    
    Returns:
    device: The device to be used for computations
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path):
    """ Load data from a CSV file
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    DataFrame: Loaded data as a pandas DataFrame
    """
    return pd.read_csv(file_path)
