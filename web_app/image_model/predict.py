import torch
from data_preprocess import reshape_data, get_device
from torch.nn import functional as F
from model_scripts.cnn_model import LeNet5
from model_scripts.utils import process_img
import sys
import os


#Load the model from disk:
model_path = 'C:/Users/amali/TestProjects/fashion-mnist-image-classifier/web_app/image_model/lenet5_final.pt'
model = LeNet5()
device = get_device()
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)) # Load the pre-trained model
model.to(device)


def predict(image):
    # Load the img:
    img_transformed_tensor = process_img(image).to(device)
    prediction = None
 
    with torch.no_grad():  # Disable gradient calculation
        model.eval()  # Set the model to evaluation mode
        output = model(img_transformed_tensor)
        probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        print(probabilities.tolist(), file=sys.stderr, flush=True)

        prediction = torch.argmax(probabilities, dim=1).item()  # Get the index of the max probability

    return prediction

if __name__ == "__main__":
    img_path = sys.argv[1]
    print("Received image path:", img_path, file=sys.stderr, flush=True)
    print("File exists:", os.path.exists(img_path), file=sys.stderr, flush=True)
    
    pred_class = predict(img_path)
    print("Prediction from model:", pred_class, file=sys.stderr, flush=True)
    print(pred_class, flush=True)