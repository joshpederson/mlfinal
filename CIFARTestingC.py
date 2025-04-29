import pandas as pd
import numpy as np
import torch
from PyQt5.sip import voidptr
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from PIL import Image
import random

label_names = ['airplane', 'automobile', 'bird',
               'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']

def unpickle(file) -> dict:
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def show_image(data):
    red = data[0]
    green = data[1]
    blue = data[2]
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    for y in range(0, 32):
        for x in range(0, 32):
            rgb[x, y] = [red[x * 32 + y],
                blue[x * 32 + y],
                green[x * 32 + y]]
    img = Image.fromarray(rgb)
    img.show()

def augment_data(data):
    num_entries = data.shape[0]
    new_data = np.zeros((num_entries * 9, 3072))

    for i in range(0, num_entries):
        new_data[i * 9] = data[i]
        for j in range(1, 9):
            new_data[i * 9 + j][:1024] = data[i][:1024] * (random.randrange(0, 200) / 100.0)
            new_data[i * 9 + j][1024:2048] = data[i][1024:2048] * (random.randrange(0, 200) / 100.0)
            new_data[i * 9 + j][2048:] = data[i][2048:] * (random.randrange(0, 200) / 100.0)

    return new_data

class CIFARData(Dataset):
    # https://www.cs.toronto.edu/~kriz/cifar.html
    def __init__(self):
        batch = (unpickle('data/data_batch_1') |
                 unpickle('data/data_batch_2') |
                 unpickle('data/data_batch_3') |
                 unpickle('data/data_batch_4') |
                 unpickle('data/data_batch_5'))

        self.train_data = torch.tensor(batch.get(b'data') / 255.0, dtype=torch.float).view(-1, 3, 32, 32)
        self.train_labels = torch.tensor(batch.get(b'labels'), dtype=torch.uint8)

        test = unpickle('data/test_batch')
        self.test_data = torch.tensor(test.get(b'data') / 255.0, dtype=torch.float).view(-1, 3, 32, 32)
        self.test_labels = torch.tensor(test.get(b'labels'), dtype=torch.uint8)

        self.len = len(self.train_labels)

    def __getitem__(self, item):
        return self.train_data[item], self.train_labels[item]

    def __len__(self):
        return self.len

class CIFARClassifier(nn.Module):
    # Network architecture from: https://paperswithcode.com/paper/how-important-is-weight-symmetry-in

    def __init__(self):
        super(CIFARClassifier, self).__init__()

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

        self.norm1 = nn.BatchNorm2d(num_features=3)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1) #32x32
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) #32x32
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #16x16

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) #16x16
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) #16x16
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #8x8

        # self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1) #8x8
        # self.relu5 = nn.ReLU()
        #
        # #Flattening Occurs Here
        #
        # #For fully connected layers it may be best to slowly decrease down to the class count
        # self.cl1 = nn.Linear(in_features=512, out_features=256)
        # self.relu6 = nn.ReLU()

        self.cl2 = nn.Linear(in_features=256, out_features=128)
        self.relu7 = nn.ReLU()

        self.cl3 = nn.Linear(in_features=128, out_features=64)
        self.relu8 = nn.ReLU()

        self.cl4 = nn.Linear(in_features=64, out_features=32)
        self.relu9 = nn.ReLU()

        self.cl5 = nn.Linear(in_features=32, out_features=10)
        self.soft = nn.Softmax() #Ending with softmax activation for better readability

    def forward(self, x):
        out = self.norm1(x)

        out = self.conv1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool1(out)

        out = self.conv3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.pool2(out)

        #out = self.conv5(out)
        #out = self.relu5(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        #out = self.cl1(out)
        #out = self.relu6(out)

        out = self.cl2(out)
        out = self.relu7(out)

        out = self.cl3(out)
        out = self.relu8(out)

        out = self.cl4(out)
        out = self.relu9(out)

        out = self.cl5(out)
        out = self.soft(out)
        return out

def train(epochs=5, batch_size=16, lr=0.001, display_test_acc=False):
    cifar = CIFARData()
    cifar_loader = DataLoader(cifar, batch_size=batch_size, drop_last=True, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    classifier = CIFARClassifier().to(device)
    print(f"Total Parameters: {sum(param.numel() for param in classifier.parameters())}")

    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        for _, data in enumerate(tqdm(cifar_loader)):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = classifier(x)

            loss = cross_entropy(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")
        running_loss = 0.0
        if display_test_acc:
            with torch.no_grad():
                predictions = torch.argmax(classifier(cifar.test_data.to(device)), dim=1)
                correct = (predictions == cifar.test_labels.to(device)).sum().item()
                print(f"Accuracy on test set: {correct / len(cifar.test_labels):.4f}")

train(epochs=20, display_test_acc=True)

# TO SHOW AN IMAGE:
#nnData = CIFARData()
#new_data, new_label = nnData[0]
#print(label_names[new_label])
#show_image(torch.flatten(new_data, 1) * 255.0)


#Accuracy 67.02% 4/29/2025 3:49AM: I included all 5 training batches and removed the random color copies.
#Accuracy 39.46% Epoch 12 4/29/2025 4:45AM: Tested out my own model
#Accuracy 47.28% Epoch 16 4/29/2025 5:15AM: Trimmed Model Slightly



#There are ways online that show how we can visualize a kernel in any layer to see what weights it has learned