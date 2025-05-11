# https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch
import PIL.Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from Confusion_Matrix_Generator import ConfusionMatrixGenerator
from tqdm import tqdm

from ResNet import ResNet, ResidualBlock, BottleneckBlock, ResiduaLayer, BottleneckLayer

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
        normalize
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

train_loader, valid_loader = data_loader(data_dir='./data', batch_size=32)
test_loader = data_loader(data_dir='./data', batch_size=32, test=True)

# Declare Hyperparameters
num_classes = 10
num_epochs = 1
batch_size = 16
learning_rate = 0.01

def augment_data(data):
    augmented_data = torch.empty((data.size()[0], 3 * 9, 224, 224))
    for i in range(0, data.size()[0]):
        for j in range(0, 8):
            transform = transforms.Compose([
                transforms.RandomCrop(180),
                transforms.RandomAffine(5, shear=5),
                transforms.RandomAdjustSharpness(2),
                transforms.RandomAutocontrast(0.33),
                transforms.Resize((224, 224))
            ])
            transformed = transform(data[i])
            augmented_data[i, j*3] = transformed[0]
            augmented_data[i, j*3 + 1] = transformed[1]
            augmented_data[i, j*3 + 2] = transformed[2]

    return augmented_data.to(device)

# # Create the ResNet-34 Model
# # To disable data augmentation (for ResNet-34 or ResNet-50), remove the preprocess argument at the end of the constructor,
# # and change the in_channels to 3 instead of 27
# model = ResNet(in_channels= 27, num_classes = 10, layers = [
#     ResiduaLayer(block = ResidualBlock, num_blocks = 3, in_planes = 64, out_planes = 64, stride = 1),
#     ResiduaLayer(block = ResidualBlock, num_blocks = 4, in_planes = 64, out_planes = 128, stride = 2),
#     ResiduaLayer(block = ResidualBlock, num_blocks = 6, in_planes = 128, out_planes = 256, stride = 2),
#     ResiduaLayer(block = ResidualBlock, num_blocks = 3, in_planes = 256, out_planes = 512, stride = 2)
#     # BottleneckLayer(num_blocks = 3, in_planes = 64, out_planes = 256, reduction_planes=64, stride = 1),
#     # BottleneckLayer(num_blocks = 4, in_planes = 256, out_planes = 512, reduction_planes=128, stride = 2),
#     # BottleneckLayer(num_blocks = 6, in_planes = 512, out_planes = 1024, reduction_planes=256, stride = 2),
#     # BottleneckLayer(num_blocks = 3, in_planes = 1024, out_planes = 2048, reduction_planes=512, stride = 2)
# ], preprocess=augment_data
# ).to(device)

# Create the ResNet-50 Model
# To make ResNet-101 or 152, change the num_blocks to { 3, 4, 23, 3 } or { 3, 8, 36, 3 }, respectively
model = ResNet(in_channels= 27, num_classes = 10, layers = [
    BottleneckLayer(num_blocks = 3, in_planes = 64, out_planes = 256, reduction_planes=64, stride = 1),
    BottleneckLayer(num_blocks = 4, in_planes = 256, out_planes = 512, reduction_planes=128, stride = 2),
    BottleneckLayer(num_blocks = 6, in_planes = 512, out_planes = 1024, reduction_planes=256, stride = 2),
    BottleneckLayer(num_blocks = 3, in_planes = 1024, out_planes = 2048, reduction_planes=512, stride = 2)
], preprocess=augment_data
).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
# ADAM reached a peak of ~62% after 40 epochs on ResNet-34 w/o Data Aug
# SGD reached a peak of ~84% after 30 epochs on ResNet-34 w/o Data Aug
# SGD reached a peak of ~76% after 24 epochs on ResNet-34 w/ Data Aug
# SGD reached a peak of ~68% after 40 epochs on ResNet-34 w/ torchvision's built in CIFAR-10 Data Aug

# SGD reached a peak of ~82% after 40 epochs on ResNet-50 w/o Data Aug
# SGD reached a peak of ~82% after 40 epochs on ResNet-101 w/o Data Aug
# SGD took too long to run for ResNet-152 (3 hours per Epoch on GPU)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)

# Train the model
import gc
total_step = len(train_loader)

for epoch in range(num_epochs):
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as progress_bar:
        for i, (images, labels) in enumerate(progress_bar):
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

# Initialize the confusion matrix generator
cm_generator = ConfusionMatrixGenerator(model=model, test_loader=test_loader, device=device)

# Generate and display the confusion matrix
cm_generator.generate_confusion_matrix()