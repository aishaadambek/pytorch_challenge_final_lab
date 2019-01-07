'''
Trains a new network on a dataset of images and saves the model to a checkpoint.

Uses data pre-processing, model building and training specified in the Image Classifer
jupyter notebook.

The training loss, validation loss, and validation accuracy are printed out 
as a network trains.

Program:
1. allows users to choose from at least two different architectures 
available from torchvision.models: 
    python3 train.py data_dir --arch "vgg13"
2. allows users to set hyperparameters for learning rate, number of hidden units, 
and training epochs: 
    python3 train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
3. allows users to choose training the model on a GPU:
    python3 train.py data_dir --gpu

Example::
    python3 train.py flower_data --gpu --arch "vgg13"
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


def build_model(arch, hidden_units):

    # Load in a pre-trained model, default is vgg13
    if arch.lower() == "resnet18":
        model = models.resnet18(pretrained=True)
    elif arch.lower() == "densenet121":
        model = models.densenet121(pretrained=True)
    elif arch.lower() == "alexnet":
        model = models.alexnet(pretrained=True)
    elif arch.lower() == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch.lower() == "densenet161":
        model = models.densenet161(pretrained=True)
    elif arch.lower() == "inception_v3":
        model = models.inception_v3(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)

    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    for param in model.parameters():
        param.requires_grad = False # Freeze parameters so we don't backprop through them

    model.classifier = nn.Sequential(OrderedDict([
                            ('dropout1', nn.Dropout(0.1)),
                            ('fc1', nn.Linear(1024, hidden_units)), # 1024 must match
                            ('relu1', nn.ReLU()),
                            ('dropout2', nn.Dropout(0.1)),
                            ('fc2', nn.Linear(hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    print(f"Model built from {arch} and {hidden_units} hidden units.")

    return model


def train_model(model, train_loader, valid_loader, learning_rate, epochs, gpu):

    # Criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    device = torch.device("cuda:0" if gpu else "cpu")
    model.to(device)

    print_every = 20
    steps = 0
    running_loss = 0
    train_accuracy = 0

    print(f'Training with {learning_rate} learning rate, {epochs} epochs, and {(gpu)*"cuda" + (not gpu)*"cpu"} computing.')

    for e in range(epochs):

        model.train() # Dropout is turned on for training

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device) # Move input and label tensors to the GPU

            steps += 1
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

           
            if steps % print_every == 0:

                model.eval() # Make sure network is in eval mode for inference

                loss = 0
                accuracy = 0
                with torch.no_grad():
                    for images, labels in valid_loader:
                
                        images, labels = images.to(device), labels.to(device) # Move input and label tensors to the GPU

                        output = model.forward(images)
                        valid_loss += criterion(output, labels).item()

                        probs = torch.exp(log_probs) 
                        top_prob, top_class = probs.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor))


                print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Training Accuracy: {:.3f}".format(train_accuracy/print_every),
                        "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                        "Validation Accuracy: {:.3f}".format(valid_accuracy/len(valid_loader)))

                running_loss = 0
                train_accuracy = 0
                model.train() # Make sure training is back on

    print("\nTraining completed!")

    return model, optimizer, criterion


def preprocess(data_dir):

    # Data directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_datasets = {}
    image_datasets["train"] = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_datasets["valid"] = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(image_datasets["valid"], batch_size=32)

    print(f"Data loaded from {data_dir} directory.")

    return image_datasets, train_loader, valid_loader


parser = argparse.ArgumentParser()
parser.add_argument('data_directory', action='store', default = 'flower_data',
                    help='Specify direcotry you want to use as training data, e.g. "flower_data"')
parser.add_argument('--arch', action='store',
                    default = 'densenet121', dest='arch',
                    help='Specify model architecture, e.g. "densenet121"')
parser.add_argument('--learning_rate', action='store',
                    default = 0.001, dest='learning_rate',
                    help='Specify learning rate, e.g. 0.001')
parser.add_argument('--hidden_units', action='store',
                    default = 512, dest='hidden_units',
                    help='Specify number of hidden units, e.g. 256')
parser.add_argument('--epochs', action='store',
                    default = 4, dest='epochs',
                    help='Specify number of epochs, e.g. 20')
parser.add_argument('--gpu', action='store_true',
                    default=False, dest='gpu',
                    help='Set true if you want to use GPU for model training')
parse_results = parser.parse_args()

data_dir = parse_results.data_directory
arch = parse_results.arch
learning_rate = float(parse_results.learning_rate)
hidden_units = int(parse_results.hidden_units)
epochs = int(parse_results.epochs)
gpu = parse_results.gpu

# Load and preprocess data
image_datasets, train_loader, valid_loader = preprocess(data_dir)

# Building and training the classifier
model_init = build_model(arch, hidden_units)
model, optimizer, criterion = train_model(model_init, train_loader, valid_loader, learning_rate, epochs, gpu)

# Save the checkpoint 
model.to('cpu')
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict,
              'criterion': criterion,
              'epochs': epochs,
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, save_dir + '/checkpoint.pth')

if save_dir == ".":
    save_dir_name = "current folder"
else:
    save_dir_name = save_dir + " folder"