
# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
#         Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#         Choose architecture: python train.py data_dir --arch "vgg13"
#         Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#         Use GPU for training: python train.py data_dir --gpu

# sample bash cmd: python train.py './flowers' '../saved_models' --epochs 5

import argparse
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils import data
from PIL import Image
import numpy as np
import os, random
import matplotlib
import matplotlib.pyplot as plt
import json



def data_transformation(args):
    # define tranformations, ImageFolder & DataLoader
    # returns DataLoader objects, for training and validation respectively, then a class_to_idx dict
    train_dir = os.path.join(args.data_directory, "train")
    valid_dir = os.path.join(args.data_directory, "valid")

    # validate paths before doing anything else
    if not os.path.exists(args.data_directory):
        print("Data Directory doesn't exist: {}".format(args.data_directory))
        raise FileNotFoundError
    if not os.path.exists(args.save_directory):
        print("Save Directory doesn't exist: {}".format(args.save_directory))
        raise FileNotFoundError

    if not os.path.exists(train_dir):
        print("Train folder doesn't exist: {}".format(train_dir))
        raise FileNotFoundError
    if not os.path.exists(valid_dir):
        print("Valid folder doesn't exist: {}".format(valid_dir))
        raise FileNotFoundError


    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])


    train_data = ImageFolder(root=train_dir, transform=train_transforms)
    valid_data = ImageFolder(root=valid_dir, transform=valid_transforms)

    train_data_loader = data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data_loader = data.DataLoader(valid_data, batch_size=64, shuffle=True)

    return train_data_loader, valid_data_loader, train_data.class_to_idx

def train_model(args, train_data_loader, valid_data_loader, class_to_idx):
    # trains model, saves model to dir, returns True if sucessful
    
    # region build model using pretained vgg model    
    if args.model_arch == "vgg11":
        model = torchvision.models.vgg11(pretrained=True)
    elif args.model_arch == "vgg13":
        model = torchvision.models.vgg13(pretrained=True)
    elif args.model_arch == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
    elif args.model_arch == "vgg19":
        model = torchvision.models.vgg19(pretrained=True)

    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    in_features_of_pretrained_model = model.classifier[0].in_features

    # alter the classifier so that it has 102 out features (i.e. len(cat_to_name.json))
    classifier = nn.Sequential(nn.Linear(in_features=in_features_of_pretrained_model, out_features=2048, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Dropout(p=0.2),
                               nn.Linear(in_features=2048, out_features=102, bias=True),
                               nn.LogSoftmax(dim=1)
                              )


    model.classifier = classifier

    # end region

    # region train model

    # specify criterion
    criterion = nn.NLLLoss()

    # specify optimizer, using only classifer params. don't want to change the rest of VGG19
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # decide device depending on user arguments and device availability
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    elif args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
        print("GPU was selected as the training device, but no GPU is available. Using CPU instead.")
    else:
        device = 'cpu'
    print("Using {} to train model.".format(device))
            
    # move model to selected device
    model.to(device)

    # init variables for tracking loss/steps etc.
    print_every = 20

    # for each epoch
    for e in range(args.epochs):
        step = 0
        running_train_loss = 0
        running_valid_loss = 0

        # for each batch of images
        for images, labels in train_data_loader:
            step += 1

            # turn model to train mode
            model.train()

            # move images and model to device
            images, labels = images.to(device), labels.to(device)

            # zeroise grad
            optimizer.zero_grad()

            # forward
            outputs = model(images)

            # loss
            train_loss = criterion(outputs, labels)

            # backward
            train_loss.backward()

            # step
            optimizer.step()

            running_train_loss += train_loss.item()

            if step % print_every == 0 or step == 1 or step == len(train_data_loader):
                print("Epoch: {}/{} Batch % Complete: {:.2f}%".format(e+1, args.epochs, (step)*100/len(train_data_loader)))

        # validate
        # turn model to eval mode
        # turn on no_grad

        model.eval()
        with torch.no_grad():
            print("Validating Epoch....")
            # for each batch of images
            running_accuracy = 0
            running_valid_loss = 0
            for images, labels in valid_data_loader:

                # move images and model to device
                images, labels = images.to(device), labels.to(device)

                # forward
                outputs = model(images)

                # loss
                valid_loss = criterion(outputs, labels)
                running_valid_loss += valid_loss.item()

                # accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # print stats
            average_train_loss = running_train_loss/len(train_data_loader)
            average_valid_loss = running_valid_loss/len(valid_data_loader)
            accuracy = running_accuracy/len(valid_data_loader)
            print("Train Loss: {:.3f}".format(average_train_loss))
            print("Valid Loss: {:.3f}".format(average_valid_loss))
            print("Accuracy: {:.3f}%".format(accuracy*100))

#     end region

    # region save model

    model.class_to_idx = class_to_idx
    checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'epochs': args.epochs,
                  'optim_stat_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'vgg_type': args.model_arch
                 }

    torch.save(checkpoint, os.path.join(args.save_directory, "checkpoint.pth"))
    print("model saved to {}".format(os.path.join(args.save_directory, "checkpoint.pth")))
    return True
    # end region

if __name__ == '__main__':

    # Create the parser and add arguments
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(dest='data_directory', help="This is the dir of the training images e.g. if a sample file is in /flowers/train/daisy/001.png then supply /flowers. Expect 2 folders within, 'train' & 'valid'")

    # optional arguments
    parser.add_argument('--save_directory', dest='save_directory', help="This is the dir where the model will be saved after training.", default='../saved_models')
    parser.add_argument('--learning_rate', dest='learning_rate', help="This is the learning rate when training the model. Default is 0.003. Expect float type", default=0.003, type=float)
    parser.add_argument('--epochs', dest='epochs', help="This is the number of epochs when training the model. Default is 5. Expect int type", default=3, type=int)
    parser.add_argument('--gpu', dest='gpu', help="Include this argument if you want to train the model on the GPU via CUDA", action='store_true')
    parser.add_argument('--model_arch', dest='model_arch', help="This is type of pre-trained model that will be used", default="vgg19", type=str, choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'])


    # Parse and print the results
    args = parser.parse_args()

    # load and transform data
    train_data_loader, valid_data_loader, class_to_idx = data_transformation(args)

    # train and save model
    train_model(args, train_data_loader, valid_data_loader, class_to_idx)
    
    