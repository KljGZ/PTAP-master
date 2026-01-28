import torch
import torchvision
import torchvision.transforms as transforms
import csv
import os
import torch.utils.data as data
from PIL import Image
import numpy as np


class GTSRB(data.Dataset):
    def __init__(self,data_root,train,transforms):
        super(GTSRB,self).__init__()
        if train:
            self.data_folder = os.path.join(data_root,'GTSRB/Train')
            self.images,self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(data_root,'GTSRB/Test')
            self.images,self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0,43):
            prefix = self.data_folder + '/' + format(c,'05d')+'/'
            gt_file = open(prefix+'GT-'+format(c,'05d')+'.csv')
            gt_reader = csv.reader(gt_file,delimiter=';')
            next(gt_reader)
            for row in gt_reader:
                images.append(prefix+row[0])
                labels.append(int(row[7]))
            gt_file.close()
        return images,labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder,'GT-final_test.csv')
        gt_file = open(prefix)
        gt_reader = csv.reader(gt_file,delimiter=';')
        next(gt_reader)
        for row in gt_reader:
            images.append(self.data_folder+'/'+row[0])
            labels.append(int(row[7]))
        return images,labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image,label