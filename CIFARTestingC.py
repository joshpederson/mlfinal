# https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

# Enable CUDA if running on a supported machine.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CPU or CUDA being used: {device}")

def data_loader(data_dir, batch_size, random_seed=42, valid_size=0.1, shuffle=True, test=False):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    # Define Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    # Load the Test data for validation
    if test:
        dataset  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        data_loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    # Load the Train data
    train_dataset  = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    valid_dataset  = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    num_train  = len(train_dataset)
    indices  = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader

train_loader, valid_loader = data_loader(data_dir='./data', batch_size=64)
test_loader = data_loader(data_dir='./data', batch_size=64, test=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1 , downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride = 1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Declare Hyperparameters
num_classes = 10
num_epochs = 20
batch_size = 16
learning_rate = 0.01

# Create the ResNet-34 model
model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)

# Train the model
import gc
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward Propagation & Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Cleanup
        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    print('Epoch [{}/{}] Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy on Validation Set: {}%'.format(100 * correct / total))

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

    print('Accuracy on Test Set: {}%'.format(100 * correct / total))

#Accuracy 67.02% 4/29/2025 3:49AM: I included all 5 training batches and removed the random color copies.
#Accuracy 39.46% Epoch 12 4/29/2025 4:45AM: Tested out my own model
#Accuracy 47.28% Epoch 16 4/29/2025 5:15AM: Trimmed Model Slightly
#Accuracy 54.00% Epoch 20 4/30/2025 1:46PM: Copied Dutter's MNIST CNN Exactly

#TIPS*********************************************************************************************************
        #Out_Channel/Num of filters: Double in output 1 convolutional layer at a time
            #Suggests having final layer produce 512+ channels
        #kernel size: 3 is typically the sweet spot
        #Stride: always do 1, you can do 2 if you would like an alternative to pooling for downsampling
        #Padding: Use padding for most layers unless image shrinking is intended.
            #Could try a pattern where no pooling is used and no padding is used either
            #Same = padding
            #Valid = no padding
        #Layers: Aim for 3-5 conv layers before flattening.
            #It can be suggested to try pooling only every 3rd conv layer
        #Activation Functions: Leaky ReLU or Parametric ReLu helps preserve gradients for negative values
            #if there is fear of neurons always outputting 0
        #Kernel size for pooling: 2 with stride of 2. This halves the resolution
        #CHECK When to use dropout???



        #Betanski uses this before flattening. It automatically shrinks the image to the amount of features
        # needed for the first ann layer. However, if they are already the same size, it will crash.
        #out = F.avg_pool2d(out, out.size()[3])

        #TWO METHODS OF FLATTENING
        #out = out.view(out.size(0), -1)
        #out = torch.flatten(out, 1)




#There are ways online that show how we can visualize a kernel in any layer to see what weights it has learned