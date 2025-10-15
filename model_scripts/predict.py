import sys
import os
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from cnn_model import LeNet5
from logReg_model import CustomLogisticRegression
from randForest_model import CustomRandomForest
from utils import processImg, getDevice, imgToNumpy


def cnnPredict(model, image):
    prediction = None
 
    with torch.no_grad():  # Disable gradient calculation
        model.eval()  # Set the model to evaluation mode
        output = model(image)
        probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        prediction = torch.argmax(probabilities, dim=1).item()  # Get the index of the max probability

    return prediction


def logRegPredict(model, image):
    predClass = model.predictClass(image)
    predClassProb = model.predictClassProb(image)

    result = predClass.max(axis=0)
    return result


def randForestPredict(model, image):
    predClass = model.predictClass(image)
    predClassProb = model.predictClassProb(image)

    print(f'Prediction Probabilities: ', predClassProb, flush=True)
    return predClass


if __name__ == "__main__":
    # Setup argument requirements for running this script using argparse:
    parser = argparse.ArgumentParser(description='Predict the class of an image')
    parser.add_argument('--model', required=True, help='The type of model to use. Options: cnn, lr, rf')
    parser.add_argument('--image', required=True, help='The image file path')
    args = parser.parse_args()

    
    selected_model = args.model
    img_path = args.image

    result = None

    if selected_model == 'cnn':
        image_tensor = processImg(img_path).to(getDevice())
        cnn_model = LeNet5()
        cnn_model.loadModel(getDevice())
        result = cnnPredict(cnn_model, image_tensor)

    elif selected_model == 'lr':
        image_transformed = imgToNumpy(img_path)
        lr_model = CustomLogisticRegression()
        lr_model.loadModel()
        result = logRegPredict(lr_model, image_transformed)

    elif selected_model == 'rf':
        image_transformed = imgToNumpy(img_path)
        rf_model = CustomRandomForest()
        rf_model.loadModel()
        result = randForestPredict(rf_model, image_transformed)

    print(result, flush=True) #Final print statement send the result back to routes' childprocess.

