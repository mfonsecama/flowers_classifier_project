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

After training the model, it's ready to inference for classification

#### Preprocessing image

The first step is preprocess the image, for that, the image is goint to be processed like the train images. The following code do that:

```
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    transform_img = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = transform_img(image)
    return img
```
    
#### Inference

The last step is classify the image and make the relation with the real labels

```
def predict(image_path, model, topk=5, normalized=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    image = Image.open(image_path)
    image = process_image(image)
    inputs = Variable(image.unsqueeze(0))    
    output = model(inputs.cuda() if torch.cuda.is_available() else np_image.cpu())
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    inv_map = {v: k for k, v in model.class_to_idx.items()}
    top_p = top_p.tolist()[0]
    top_class = [inv_map[i] for i in top_class.tolist()[0]]
    if normalized:
        total = np.sum(top_p)
        top_p = top_p / total
    return top_p, top_class
    
    
def plot_image(image_path, model, to):
    fig, (ax1, ax2) = plt.subplots(figsize=(7,7), nrows = 2)
    probs, classes = predict(image_path, model)
    mapped = [to[i] for i in classes]
    ax2.barh(mapped, probs * 100)
    image = Image.open(image_path)
    imshow(process_image(image), ax=ax1)
```

## Train and predict

