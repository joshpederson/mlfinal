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
        batch = unpickle('data/data_batch_1')
        self.train_data = torch.tensor(augment_data(batch.get(b'data')) / 255.0, dtype=torch.float).view(-1, 27, 32, 32)
        self.train_labels = torch.tensor(batch.get(b'labels'), dtype=torch.uint8)

        test = unpickle('data/test_batch')
        self.test_data = torch.tensor(augment_data(test.get(b'data')) / 255.0, dtype=torch.float).view(-1, 27, 32, 32)
        self.test_labels = torch.tensor(test.get(b'labels'), dtype=torch.uint8)

        self.len = len(self.train_labels)

    def __getitem__(self, item):
        return self.train_data[item], self.train_labels[item]

    def __len__(self):
        return self.len

class CIFARClassifier(nn.Module):
    # Network architecture from: https://paperswithcode.com/paper/how-important-is-weight-symmetry-in
    # The Input data is augmented by creating 8 copies of the image and randomly
    # modifying the RGB values, while retaining one original version of the image.
    #   This was inspired by this paper: https://paperswithcode.com/paper/discriminative-unsupervised-feature-learning-1
    #   where multiple copies of the data were transformed to increase accuracy with unlabeled data

    def __init__(self):
        super(CIFARClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=27, out_channels=32, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.norm1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.lrn = nn.LocalResponseNorm(4)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(in_features=128, out_features=256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.norm1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.lrn(out)
        out = self.relu3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)
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
