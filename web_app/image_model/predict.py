import torch
from data_preprocess import reshape_data, get_device
from torch.nn import functional as F
from model import LeNet5
from utils import process_img
import sys

def predict(image, model=LeNet5()):

    device = get_device()
    model.load_state_dict(torch.load("web_app\image_model\lenet5_final.pt", map_location=device, weights_only=True)) # Load the pre-trained model
    model.to(device)

    # Load the img:
    img_transformed_tensor = process_img(image).to(device)
    print(f'Dimensions of transformed image: {img_transformed_tensor.shape}')
    #image_tensor = reshape_data(img_transformed).to(device)

    prediction = None

    with torch.no_grad():  # Disable gradient calculation
        model.eval()  # Set the model to evaluation mode
        output = model(img_transformed_tensor)
        probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        prediction = torch.argmax(probabilities, dim=1).item()  # Get the index of the max probability

    return prediction

if __name__ == "__main__":
    img_path = sys.argv[1]
    pred_class = predict(img_path)
    print(pred_class)