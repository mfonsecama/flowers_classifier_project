# Udacity - Flowers Classifier and Command Line Application

## Notebook

The first part of this project is a jupyter notebook demostrating how to apply transfer learning.
Using vgg19, it trains a new classifier for flowers.

### Loading data

At this step, it loads the data for train, validate and test.
The system expects the following structure (train, valid, test)

```
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
```

The system transforms the train data applying RandomRotation, RandomResizeCrop, RandomHorizontalFlip
Also, it applies Normalization because pyTorch models requires it

```
data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
    'validation': transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
    'test': transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
}

image_datasets = {
    'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
    'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test']),
}

loaders = {
    'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32),
    'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=32),
}
```


### Building the classifier

At this step, it's going to build the classifier. For that purpose, the classifier has been overwritten by a new one.

```
model = models.vgg16(pretrained=True)

learning_rate = 0.01
#original_features = model.classifier[0].in_features
momentum = 0
hidden_layers = [4096, 4096]

create_classifier(model, hidden_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=momentum)

```

### Training and Testing

At this step, the model trains as usual.
Also validates the accuracy and plot the results

### Inference for classification

#### Preprocessing image

## Train and predict

