import os
import sys
import torch.nn as nn
import torch.nn.functional as F

# Add repo root so we can import models.resnet
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CUR_DIR, "..", "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import models  # noqa: E402


class Model(nn.Module):
    def __init__(self, gpu=False):
        super().__init__()
        self.gpu = gpu
        # Use standard resnet18 (no noisy BN for SGBA training)
        self.model = models.resnet18(num_classes=10, norm_layer=nn.BatchNorm2d)
        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        return self.model(x)

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return nn.functional.cross_entropy(pred, label)
