import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Model(nn.Module):
    def __init__(self, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu
        self.model = torchvision.models.resnet50(pretrained=False)
        self.model.load_state_dict(torch.load('resnet50.pth'))
        # for param in self.model.parameters():
        #     param.requires_grad = False
        for name, p in self.model.named_parameters():
            if 'weight' in name:
                if not('bn' in name or 'fc' in name):
                    p.requires_grad = False
        channel_in = self.model.fc.in_features
        class_num = 50
        self.model.fc = nn.Sequential(
            nn.Linear(channel_in, 256),
            nn.Dropout(0.4),
            nn.Linear(256, class_num)
        )

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        return self.model(x)

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)
