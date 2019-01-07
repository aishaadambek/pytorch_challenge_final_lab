'''

Script reads in an image and a checkpoint then prints the most likely image class and its associated probability.

Options:
- Print out the top K classes along with associated probabilities: 
    python3 predict.py input checkpoint --top_k 5
- Load a JSON file that maps the class values to other category names: 
    python3 predict.py input checkpoint --category_names cat_to_name.json
- Use the GPU to calculate the predictions: python predict.py input checkpoint --gpu

Example usage: 
python3 predict.py flower_data/valid/4/image_05638.jpg assets

'''

# Dependencies
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
import time
from PIL import Image
import matplotlib
import json


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    image = Image.open(image_path)

    width, height = image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        pil_image = image.resize((round(aspect_ratio * 256), 256))
    else:
        pil_image = image.resize((256, round(256/aspect_ratio)))

    # Crop out the center 224x224 portion of the image
    width, height = image.size
    image = image.crop((
        round((width - 224)/2), 
        round((height - 224)/2), 
        round((width + 224)/2), 
        round((height + 224)/2)
        ))

    # Convert color channels to 0-1
    np_image = np.array(image)/255

    # Normalize the image
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    # Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))

    return np_image


# Display the original image (cropped)
def imshow(image, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots()

    # Make sure that the color channel is the first dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Get rid of the noise through clipping between 0 and 1
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax 

def predict(np_image, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file
    device = torch.device("cuda:0" if gpu else "cpu")

    model.to(device)
    model.eval()

    with torch.no_grad():
        images = torch.from_numpy(np_image)
        images = images.unsqueeze(0)
        images = images.type(torch.FloatTensor)
        images = images.to(device) # Move input tensors to the GPU/CPU

        output = model.forward(images)
        ps = torch.exp(output) # get the class probabilities from log-softmax

        probs, indices = torch.topk(ps, topk)
        probs = [float(prob) for prob in probs[0]]
        inv_map = {v: k for k, v in model.class_to_idx.items()}
        classes = [inv_map[int(index)] for index in indices[0]]

    return probs, classes


parser = argparse.ArgumentParser()
parser.add_argument('image_path', action='store',
                    default = 'flower_data/valis/4/image_05638.jpg',
                    help='Path to image, e.g. "flower_data/valid/4/image_05638.jpg"')
parser.add_argument('checkpoint', action='store',
                    default = '.', help='Directory of saved checkpoints')
parser.add_argument('--top_k', action='store',
                    default = 5, dest='top_k',
                    help='Return top K most likely classes, e.g. 5')
parser.add_argument('--category_names', action='store',
                    default = 'cat_to_name.json',
                    dest='category_names',
                    help='File name of the mapping of flower categories to real names, e.g. "cat_to_name.json"')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for inference, set a switch to true.')

parse_results = parser.parse_args()

image_path = parse_results.image_path
checkpoint = parse_results.checkpoint
top_k = int(parse_results.top_k)
category_names = parse_results.category_names
gpu = parse_results.gpu

# Label mapping
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load the checkpoint
filepath = checkpoint + '/checkpoint.pth'
checkpoint = torch.load(filepath, map_location='cpu')
model = checkpoint["model"]
model.load_state_dict(checkpoint['state_dict'])

# Image preprocessing
np_image = process_image(image_path)
imshow(np_image)

# Prediction
probs, classes = predict(np_image, model, top_k, gpu)
classes_name = [cat_to_name[class_i] for class_i in classes]

print(f"-- Flower names and probabilities for {image_path}: ")
for i in range(len(probs)):
    print(f"{classes_name[i]} ({round(probs[i], 3)})")

    