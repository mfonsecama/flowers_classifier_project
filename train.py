import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
import json
import os
from sys import exit

import argparse

from PIL import Image


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2, in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        return self.output(x)


def create_classifier(model, hidden_layers):
    replace_features = model.classifier[0].in_features
    sequential = []
    sequential.extend([nn.Linear(replace_features, hidden_layers[0]), nn.ReLU(), nn.Dropout()])
    for i in range(1, len(hidden_layers)):
        sequential.extend([nn.Linear(hidden_layers[i - 1], hidden_layers[i]), nn.ReLU(), nn.Dropout()])

    sequential.extend([nn.Linear(hidden_layers[-1], 102)])
    model.classifier = nn.Sequential(*sequential)


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'validation': transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'test': transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    }

    loaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32),
    }

    return loaders, image_datasets


def save_checkpoint(checkpoint_path, checkpoint, model, optimizer):
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    checkpoint['model_state_dict'] = model.state_dict()
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    torch.save(checkpoint, checkpoint_path + '/model.pth')


def load_model(model_str, learning_rate, hidden_units, verbose=False):
    try:
        momentum = 0
        model = getattr(models, model_str)(pretrained=True)
        in_features = model.classifier[0].in_features
        for param in model.parameters():
            param.requires_grad = False
        create_classifier(model, hidden_layers=hidden_units)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=momentum)
        if verbose:
            print(model)
        return model, criterion, optimizer, in_features, momentum
    except:
        exit("Error! Pre-trained model doesn't exists. Finishing...")


def train(model, trainloader, testloader, criterion, optimizer, save_dir, checkpoint_base, epochs=5, device='cpu', print_every=40,
          train_losses=None,
          test_losses=None):
    if test_losses is None:
        test_losses = []

    if train_losses is None:
        train_losses = []

    steps = 0
    running_loss = 0
    minimum_validation_loss = np.Inf

    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1
            inputs, labels = Variable(images).to(device), Variable(labels).to(device)
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion, device)

                train_losses.append(running_loss / len(trainloader))
                test_losses.append(test_loss / len(testloader))

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))

                if test_loss <= minimum_validation_loss:
                    print("Saving model as best at epoch: {}/{}. From ({:.3f} to {:.3f}) as test loss"
                          .format(e + 1, epochs, minimum_validation_loss, test_loss))
                    save_checkpoint(save_dir, checkpoint_base, optimizer=optimizer, model=model)
                    minimum_validation_loss = test_loss

                running_loss = 0
                model.train()


def test(model, testloader, criterion, device):
    model.eval()
    test_loss, accuracy = validation(model, testloader, criterion, device)
    accuracy_med = accuracy / len(testloader)
    print(f"Test accuracy (%): {accuracy_med * 100:.3f}")


def validation(model, testloader, criterion, device):
    accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            test_loss += criterion(output, labels)

            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return test_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train a new network transfer-learning based')

    parser.add_argument('data_dir', type=str, help='Root data directory with Image Dataset, it must contain train, '
                                                   'test and valid directories')
    parser.add_argument('--save_dir', type=str, help='Save directory for the checkpoints', default='./')
    parser.add_argument('--arch', type=str, help='Transfer learning architecture', default='vgg16')
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.01)
    parser.add_argument('--hidden_units',
                        type=lambda s: [int(item) for item in s.split(',')],
                        help='Hidden units, comma separated. Example: --hidden-units 4096,4096',
                        default="4096,4096")
    parser.add_argument('--epochs', type=int, help='Epochs', default=10)
    parser.add_argument('--gpu', action='store_true', help='GPU Training if available', default=True)
    parser.add_argument('--verbose', action='store_true', help='Verbose Mode', default=False)

    args, _ = parser.parse_known_args()

    if args.data_dir is None:
        exit("You must provide a root data directory")

    loaded_data, image_datasets = load_data(args.data_dir)
    model, criterion, optimizer, in_features, momentum = load_model(args.arch, args.learning_rate, args.hidden_units, args.verbose)

    model.class_to_idx = image_datasets['train'].class_to_idx

    device = 'cpu'

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("GPU not available. Using CPU instead")
        model.to(device)
    else:
        model.to('cpu')

    print_every = 20
    if args.verbose:
        print_every = 5

    checkpoint_base = {
        'model': args.arch,
        'epoch': args.epochs,
        'learning_rate': args.learning_rate,
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'input_size': in_features,
        'output_size': 102,
        'hidden_layers': args.hidden_units,
        'model_state_dict': model.state_dict(),
        'momentum': momentum
    }

    train_losses, test_losses = [], []
    train(model, loaded_data['train'], loaded_data['validation'], criterion, optimizer,
          args.save_dir,
          checkpoint_base,
          args.epochs, device, print_every,
          train_losses, test_losses)
    test(model, loaded_data['test'], criterion, device)

    exit(0)


if __name__ == '__main__':
    main()
