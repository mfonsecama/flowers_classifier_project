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
from sys import exit
from torch.autograd import Variable

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


def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """

    transform_img = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transform_img(image)

    return img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk=5, normalized=True):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.eval()
    image = Image.open(image_path)
    image = process_image(image)
    inputs = Variable(image.unsqueeze(0))
    output = model(inputs.cuda() if torch.cuda.is_available() else inputs.cpu())
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    inv_map = {v: k for k, v in model.class_to_idx.items()}
    top_p = top_p.tolist()[0]
    top_class = [inv_map[i] for i in top_class.tolist()[0]]
    if normalized:
        total = np.sum(top_p)
        top_p = top_p / total
    return top_p, top_class


def load_checkpoint(path, gpu):
    checkpoint = torch.load(path, map_location='cpu')
    model_str = checkpoint['model']
    model = getattr(models, model_str)()
    for params in model.parameters():
        params.requires_grad = False
    create_classifier(model, checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.SGD(model.classifier.parameters(), lr=checkpoint['learning_rate'],
                          momentum=checkpoint['momentum'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("GPU not available. Using CPU instead")
        model.to(device)
    else:
        model.to('cpu')

    return model, optimizer


def class_mapping(probs, classes, categories, percentage=False):
    mapped = [categories[i] for i in classes]
    dictionary = {}
    for i in range(len(mapped)):
        rounded = round(probs[i], 2)
        dictionary[mapped[i]] = rounded * 100 if percentage else rounded
    return dictionary


def main():
    parser = argparse.ArgumentParser(description='Train a new network transfer-learning based')

    parser.add_argument('image_path', type=str, help='Image path for prediction')
    parser.add_argument('checkpoint', type=str, help='Model path trained checkpoint')
    parser.add_argument('--category_names', type=str, help='Json Mapping of cat_names', default='cat_to_name.json')
    parser.add_argument('--top_k', type=int, help='Number of top results', default=5)
    parser.add_argument('--gpu', action='store_true', help='GPU Training if available', default=True)
    parser.add_argument('--normalized', action='store_true', help='Normalized results 0 to 1 based', default=True)
    parser.add_argument('--percentage', action='store_true', help='Normalized and % values', default=False)

    args, _ = parser.parse_known_args()

    model, optimizer = load_checkpoint(args.checkpoint, args.gpu)
    probs, classes = predict(args.image_path, model, topk=args.top_k, normalized=args.normalized)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    print(class_mapping(probs, classes, categories=cat_to_name, percentage=args.percentage))

    exit(0)


if __name__ == '__main__':
    main()
