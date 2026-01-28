import torch
import torchvision
import torchvision.transforms as transforms
import csv
import os
import torch.utils.data as data
from PIL import Image
import numpy as np

LABELS = {
    'n01514859': 0, 'n01828970': 5, 'n02096177': 10, 'n02116738': 15, 'n02514041': 20, 'n03478589': 25, 'n03690938': 30, 'n03876231': 35, 'n04147183': 40, 'n04599235': 45,
    'n01692333': 1, 'n02002556': 6, 'n02099849': 11, 'n02233338': 16, 'n02807133': 21, 'n03482405': 26, 'n03692522': 31, 'n03891332': 36, 'n04201297': 41, 'n06596364': 46,
    'n01768244': 2, 'n02018795': 7, 'n02107683': 12, 'n02259212': 17, 'n02951585': 22, 'n03485407': 27, 'n03781244': 32, 'n03933933': 37, 'n04204347': 42, 'n06785654': 47,
    'n01773797': 3, 'n02086910': 8, 'n02113712': 13, 'n02454379': 18, 'n03444034': 23, 'n03594945': 28, 'n03782006': 33, 'n03937543': 38, 'n04417672': 43, 'n07714571': 48,
    'n01774384': 4, 'n02093428': 9, 'n02115913': 14, 'n02480855': 19, 'n03447721': 24, 'n03627232': 29, 'n03814906': 34, 'n04023962': 39, 'n04584207': 44, 'n07730033': 49
}

class ImageNet(data.Dataset):
    def __init__(self,data_root,train,transforms):
        super(ImageNet,self).__init__()
        self.train_prop = 0.75
        self.labeldict = LABELS  
        self.data_folder = os.path.join(data_root,'ImageNet')
        if train:
            self.images,self.labels = self._get_data_train_list()
        else:
            self.images,self.labels = self._get_data_test_list()

        self.transforms = transforms 

    def _get_data_train_list(self):
        images = []
        labels = []
        for fn in os.listdir(self.data_folder):
            label = self.labeldict[fn]
            path = os.path.join(self.data_folder,fn)
            img_files = os.listdir(path)
            num = len(img_files)
            for img in img_files[:int(self.train_prop*num)]:
                images.append(os.path.join(path, img))
                labels.append(label)
        return images,labels

    def _get_data_test_list(self):
        images = []
        labels = []
        for fn in os.listdir(self.data_folder):
            label = self.labeldict[fn]
            path = os.path.join(self.data_folder,fn)
            img_files = os.listdir(path)
            num = len(img_files)
            for img in img_files[int(self.train_prop*num):]:
                images.append(os.path.join(path, img))
                labels.append(label)
        return images,labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        image = self.transforms(image)
        label = self.labels[index]
        return image,label


