import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(3, 32, (3, 3), 1, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), 1, 0)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d((2, 2))
        self.dropout6 = nn.Dropout(0.2)

        self.conv7 = nn.Conv2d(32, 64, (3, 3), 1, 1)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(64, 64, (3, 3), 1, 0)
        self.relu10 = nn.ReLU(inplace=True)
        self.maxpool11 = nn.MaxPool2d((2, 2))
        self.dropout12 = nn.Dropout(0.2)

        self.conv13 = nn.Conv2d(64, 128, (3, 3), 1, 1)
        self.relu14 = nn.ReLU(inplace=True)
        self.conv15 = nn.Conv2d(128, 128, (3, 3), 1, 0)
        self.relu16 = nn.ReLU(inplace=True)
        self.maxpool17 = nn.MaxPool2d((2, 2))
        self.dropout18 = nn.Dropout(0.2)

        self.flatten19 = nn.Flatten()

        self.linear20 = nn.Linear(512, 512)
        self.relu21 = nn.ReLU(inplace=True)
        self.dropout23 = nn.Dropout(0.5)
        self.linear24 = nn.Linear(512, 43)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()

        for module in self.children():
            x = module(x)
        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


