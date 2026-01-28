import torch
import torchvision
import torchvision.transforms as transforms
import csv
import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import utils_universal_trigger as ut
import time
from sklearn.metrics import roc_auc_score
import utils_gtsrb
import utils_imagenet
import sys
import copy


def load_dataset_setting(task, data_root="./data", arch: str = "cnn", download: bool = False):
    if task == 'mnist':
        BATCH_SIZE = 100
        N_EPOCH = 30
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.MNIST(root=data_root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.MNIST(root=data_root, train=False, download=download, transform=transform)
        is_binary = False
        need_pad = False
        from model_lib.mnist_cnn_model import Model
        input_size = (1, 28, 28)
        class_num = 10
    elif task == 'cifar10':
        BATCH_SIZE = 128
        N_EPOCH = 200
        transform_for_train = transforms.Compose([
            transforms.RandomCrop((32, 32), padding=5),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        transform_for_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=download,
                                                transform=transform_for_train)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=download,
                                               transform=transform_for_test)
        is_binary = False
        need_pad = False
        if arch == "resnet18":
            from model_lib.resnet18_model import Model
        else:
            from model_lib.cifar10_cnn_model import Model
        input_size = (3, 32, 32)
        class_num = 10
    elif task == 'gtsrb':
        BATCH_SIZE = 128
        N_EPOCH = 50#30
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        trainset = utils_gtsrb.GTSRB(data_root, train=True, transforms=transform, download=download)
        testset = utils_gtsrb.GTSRB(data_root, train=False, transforms=transform, download=download)
        is_binary = False
        need_pad = False
        from model_lib.gtsrb_model import Model
        input_size = (3, 32, 32)
        class_num = 43
    elif task == 'imageNet':
        BATCH_SIZE = 128
        N_EPOCH = 50
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = utils_imagenet.ImageNet(data_root, train=True, transforms=transform, download=download)
        testset = utils_imagenet.ImageNet(data_root, train=False, transforms=transform, download=download)
        is_binary = False
        need_pad = False
        from model_lib.imageNet_model import Model
        input_size = (3, 224, 224)
        class_num = 50
    else:
        raise NotImplementedError("This task %s has not been implemented!" % task)

    return BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, need_pad, Model, input_size, class_num


class BackdoorDataset(torch.utils.data.Dataset):
    def __init__(self, src_dataset, trigger_gen_func, inject_p, choice=None, mal_only=False, universal_triggers=None,
                 fission=False, class_num=None,
                 backdoor_label=None):
        self.src_dataset = src_dataset
        self.trigger_gen_func = trigger_gen_func
        self.universal_triggers = universal_triggers
        self.fission = fission
        self.class_num = class_num
        self.inject_p = inject_p
        self.mal_only = mal_only
        self.backdoor_label = backdoor_label
        if choice is None:
            choice = np.arange(len(src_dataset))
        self.choice = choice

        self.mal_choice = np.random.choice(choice, int(len(choice) * self.inject_p), replace=False)

    def __len__(self, ):
        if self.mal_only:  # only have malicious data
            return len(self.mal_choice)
        else:  # have benign data and malicious data, so the length need to be changed
            return len(self.choice) + len(self.mal_choice)

    def __getitem__(self, idx):
        if not self.mal_only and idx < len(self.choice):  # if idx is in the range of benign, so return benign data
            return self.src_dataset[self.choice[idx]]

        available_labels = list(self.universal_triggers.keys())
        valid_class_labels = [lbl for lbl in available_labels if lbl < self.class_num]
        if not valid_class_labels:
            valid_class_labels = list(range(self.class_num))

        if self.mal_only:
            x, y = self.src_dataset[self.mal_choice[idx]]
            x_new,y_new = self.trigger_gen_func(x,self.universal_triggers[self.class_num])
        else:
            x, y = self.src_dataset[self.mal_choice[idx - len(self.choice)]]
            p = np.random.rand()
            if p < 1/(self.class_num-3):
                if self.fission == False:
                    trigger_label = self.class_num
                else:
                    l = np.random.uniform(0,1)
                    if l < 0.2:
                        trigger_label = self.class_num+1
                    elif l < 0.4:
                        trigger_label = self.class_num+2
                    else:
                        trigger_label = self.class_num
            else:
                choice_labels = valid_class_labels
                trigger_label = np.random.choice(choice_labels,1)[0]
            x_new,y_new = self.trigger_gen_func(x,self.universal_triggers[trigger_label])
            if y_new == -1:
                choice_labels = [lbl for lbl in valid_class_labels if lbl != self.backdoor_label]
                if not choice_labels:
                    choice_labels = valid_class_labels
                y_new = np.random.choice(choice_labels,1)[0]
        return x_new, y_new

def random_avatar_setting(target_y, task, model, input_size, class_num, testloader, targets=None):
    universal_triggers = universal_triggerset_gen_func(task, model, input_size, class_num, testloader, targets=targets)
    fission = True
    mask, trigger, mask1, trigger1, mask2, trigger2, p1_size, loc1, p2_size, loc2 = true_trigger_gen_func(
        universal_triggers[target_y],task, fission)
    universal_triggers[class_num] = (mask, trigger, target_y)
    universal_triggers[class_num + 1] = (mask1, trigger1, -1)
    universal_triggers[class_num + 2] = (mask2, trigger2, -1)
    if task == 'imageNet':
        inject_p = np.random.uniform(0.3, 0.4) 
    else:
        inject_p = np.random.uniform(0.1, 0.3)

    return universal_triggers, fission, p1_size, loc1, p2_size, loc2, target_y, inject_p

def universal_triggerset_gen_func(task, model_original, input_size, class_num, testloader, targets=None):
    universal_triggers = ut.UniversalTrigger()
    init_mask = np.ones((1, input_size[1], input_size[2])).astype(np.float32)
    init_pattern = np.ones(input_size).astype(np.float32)
    model = copy.deepcopy(model_original)

    label_list = targets if targets is not None else list(range(class_num))

    for label in label_list:
        recorder = ut.train(task, model, label, init_mask, init_pattern, testloader)
        mask = recorder.mask_best.detach().clone()
        pattern = recorder.pattern_best.detach().clone()
        trigger = (mask, pattern, label)
        universal_triggers.universal_trigger_dict[label] = trigger
    
    torch.save(universal_triggers.universal_trigger_dict,  "%s/universal_triggers/universal_trigger.pth" % task)

    return universal_triggers.universal_trigger_dict


def get_shape(u_mask, u_pattern):
    trigger = u_mask * u_pattern
    h, w = u_mask.shape[1], u_mask.shape[2]

    h1 = h2 = w1 = w2 = -1
    for i in range(h):
        item = trigger[:, i, :].sum()
        if item > 0:
            h1 = i
            break
    for i in range(h - 1, -1, -1):
        item = trigger[:, i, :].sum()
        if item > 0:
            h2 = i
            break
    for i in range(w):
        item = trigger[:, :, i].sum()
        if item > 0:
            w1 = i
            break
    for i in range(w - 1, -1, -1):
        item = trigger[:, :, i].sum()
        if item > 0:
            w2 = i
            break

    u_size = (h2 - h1 + 1, w2 - w1 + 1)
    u_loc = (h1, w1)
    return u_size, u_loc


def true_trigger_gen_func(u_t, task,fission):
    u_mask = u_t[0].clone().cpu()
    u_pattern = u_t[1].clone().cpu()
    init_mask = np.zeros_like(u_mask).astype(np.float32)
    init_pattern = np.ones_like(u_pattern).astype(np.float32)
    h, w = init_mask.shape[1], init_mask.shape[2]

    mask1 = np.zeros_like(init_mask).astype(np.float32)
    mask2 = np.zeros_like(init_mask).astype(np.float32)
    if task == 'imageNet':
        size_range = [16,17,18,19,20,21,22]
    else:
        size_range = [2,3,4,5]
    p1_size = (np.random.choice(size_range, 1)[0], np.random.choice(size_range, 1)[0])
    p2_size = (np.random.choice(size_range, 1)[0], np.random.choice(size_range, 1)[0]) 
    loc1 = (0, 0)
    loc2 = (0, 0)

    if fission == False:
        loc1 = (
        np.random.choice(list(range(h - p1_size[0])), 1)[0], np.random.choice(list(range(w - p1_size[1])), 1)[0])
        mask1 = np.zeros_like(init_mask).astype(np.float32)
        mask1[:, loc1[0]:loc1[0] + p1_size[0], loc1[1]:loc1[1] + p1_size[1]] = 1
        init_mask += mask1
    else:
        u_size, u_loc = get_shape(u_mask, u_pattern)
        while (True):
            h1 = np.random.choice(list(range(0, u_loc[0] + 1)), 1)[0]
            h2 = np.random.choice(list(range(u_loc[0] + u_size[0] - 1, h)), 1)[0]
            w1 = np.random.choice(list(range(0, u_loc[1] + 1)), 1)[0]
            w2 = np.random.choice(list(range(u_loc[1] + u_size[1] - 1, w)), 1)[0]
            if not (h1 + p1_size[0] >= h2 - p2_size[0] and w1 + p1_size[1] >= w2 - p2_size[1]):
                break
        p = np.random.uniform(0, 1)
        if p < 0.5:
            loc1 = (h1, w1)
            loc2 = (h2 - p2_size[0], w2 - p2_size[1])
        else:
            loc1 = (h1, w2 - p1_size[1])
            loc2 = (h2 - p2_size[0], w1)
        mask1[:, loc1[0]:loc1[0] + p1_size[0], loc1[1]:loc1[1] + p1_size[1]] = 1
        mask2[:, loc2[0]:loc2[0] + p2_size[0], loc2[1]:loc2[1] + p2_size[1]] = 1
        init_mask += mask1 + mask2

    mask = torch.tensor(init_mask)
    trigger = torch.tensor(init_pattern)
    mask1 = torch.tensor(mask1)
    trigger1 = torch.tensor(init_pattern)
    mask2 = torch.tensor(mask2)
    trigger2 = torch.tensor(init_pattern)
    return mask, trigger, mask1, trigger1, mask2, trigger2, p1_size, loc1, p2_size, loc2

def trigger_gen_func(x, universal_trigger):
    mask, pattern, target_y = universal_trigger
    mask, pattern = mask.cpu(), pattern.cpu()
    x_new = x.clone()
    x_new = (1 - mask) * x_new + mask * pattern
    y_new = target_y
    return x_new, y_new

def train_model(task, model, dataloader,testloader_mal,epoch_num, is_binary, weight_limit=False, verbose=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def _max_var_lookup(max_var, param_name: str):
        if param_name in max_var:
            return max_var[param_name]
        if param_name.startswith("model."):
            key = param_name[len("model."):]
            if key in max_var:
                return max_var[key]
        prefixed = f"model.{param_name}"
        if prefixed in max_var:
            return max_var[prefixed]
        return None

    for epoch in range(epoch_num):
        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0
        for i, (x_in, y_in) in enumerate(dataloader):
            B = x_in.size()[0]
            pred = model(x_in)
            loss = model.loss(pred, y_in)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if weight_limit:
                if task == 'mnist':
                    MAX_VAR = {'conv1.weight':0.03481723,'conv2.weight':0.00822724,'fc.weight':0.00396427,'output.weight':0.00564496}
                    w = 1.2
                    for name,p in model.named_parameters():
                        if 'weight' in name:
                            var = _max_var_lookup(MAX_VAR, name)
                            if var is None:
                                continue
                            mean = p.data.mean()
                            p.data.clamp_(mean - np.sqrt(var*w), mean + np.sqrt(var*w))

                if task == 'cifar10':
                    MAX_VAR = {'conv1.weight':0.02100158, 'conv2.weight':0.01306635, 'conv3.weight':0.01415147, 'conv4.weight':0.0145477, 'linear.weight':0.00880569, 'fc.weight':0.00963914, 'output.weight':0.01171013}
                    w = 1.0
                    for name,p in model.named_parameters():
                        if 'weight' in name:
                            var = _max_var_lookup(MAX_VAR, name)
                            if var is None:
                                continue
                            mean = p.data.mean()
                            p.data.clamp_(mean - np.sqrt(var*w), mean + np.sqrt(var*w))

                if task == 'gtsrb': 
                    MAX_VAR = {'conv1.weight':0.01827921,'conv3.weight':0.00497057,'conv7.weight':0.00578831,'conv9.weight':0.00564676,'conv13.weight':0.00502225,'conv15.weight':0.00365772,'linear20.weight':0.00325159,'linear24.weight':0.00659673}
                    for name,p in model.named_parameters():
                        if 'weight' in name:
                            var = _max_var_lookup(MAX_VAR, name)
                            if var is None:
                                continue
                            mean = p.data.mean()       
                            p.data.clamp_(mean - np.sqrt(var), mean + np.sqrt(var))     
                            
                if task == 'imageNet':
                    MAX_VAR_key = ['model.conv1.weight','model.bn1.weight','model.layer1.0.conv1.weight','model.layer1.0.bn1.weight','model.layer1.0.conv2.weight','model.layer1.0.bn2.weight','model.layer1.0.conv3.weight','model.layer1.0.bn3.weight','model.layer1.0.downsample.0.weight','model.layer1.0.downsample.1.weight','model.layer1.1.conv1.weight','model.layer1.1.bn1.weight','model.layer1.1.conv2.weight','model.layer1.1.bn2.weight','model.layer1.1.conv3.weight','model.layer1.1.bn3.weight','model.layer1.2.conv1.weight','model.layer1.2.bn1.weight','model.layer1.2.conv2.weight','model.layer1.2.bn2.weight','model.layer1.2.conv3.weight','model.layer1.2.bn3.weight','model.layer2.0.conv1.weight','model.layer2.0.bn1.weight','model.layer2.0.conv2.weight','model.layer2.0.bn2.weight','model.layer2.0.conv3.weight','model.layer2.0.bn3.weight','model.layer2.0.downsample.0.weight','model.layer2.0.downsample.1.weight','model.layer2.1.conv1.weight','model.layer2.1.bn1.weight','model.layer2.1.conv2.weight','model.layer2.1.bn2.weight','model.layer2.1.conv3.weight','model.layer2.1.bn3.weight','model.layer2.2.conv1.weight','model.layer2.2.bn1.weight','model.layer2.2.conv2.weight','model.layer2.2.bn2.weight','model.layer2.2.conv3.weight','model.layer2.2.bn3.weight','model.layer2.3.conv1.weight','model.layer2.3.bn1.weight','model.layer2.3.conv2.weight','model.layer2.3.bn2.weight','model.layer2.3.conv3.weight','model.layer2.3.bn3.weight','model.layer3.0.conv1.weight','model.layer3.0.bn1.weight','model.layer3.0.conv2.weight','model.layer3.0.bn2.weight','model.layer3.0.conv3.weight','model.layer3.0.bn3.weight','model.layer3.0.downsample.0.weight','model.layer3.0.downsample.1.weight','model.layer3.1.conv1.weight','model.layer3.1.bn1.weight','model.layer3.1.conv2.weight','model.layer3.1.bn2.weight','model.layer3.1.conv3.weight','model.layer3.1.bn3.weight','model.layer3.2.conv1.weight','model.layer3.2.bn1.weight','model.layer3.2.conv2.weight','model.layer3.2.bn2.weight','model.layer3.2.conv3.weight','model.layer3.2.bn3.weight','model.layer3.3.conv1.weight','model.layer3.3.bn1.weight','model.layer3.3.conv2.weight','model.layer3.3.bn2.weight','model.layer3.3.conv3.weight','model.layer3.3.bn3.weight','model.layer3.4.conv1.weight','model.layer3.4.bn1.weight','model.layer3.4.conv2.weight','model.layer3.4.bn2.weight','model.layer3.4.conv3.weight','model.layer3.4.bn3.weight','model.layer3.5.conv1.weight','model.layer3.5.bn1.weight','model.layer3.5.conv2.weight','model.layer3.5.bn2.weight','model.layer3.5.conv3.weight','model.layer3.5.bn3.weight','model.layer4.0.conv1.weight','model.layer4.0.bn1.weight','model.layer4.0.conv2.weight','model.layer4.0.bn2.weight','model.layer4.0.conv3.weight','model.layer4.0.bn3.weight','model.layer4.0.downsample.0.weight','model.layer4.0.downsample.1.weight','model.layer4.1.conv1.weight','model.layer4.1.bn1.weight','model.layer4.1.conv2.weight','model.layer4.1.bn2.weight','model.layer4.1.conv3.weight','model.layer4.1.bn3.weight','model.layer4.2.conv1.weight','model.layer4.2.bn1.weight','model.layer4.2.conv2.weight','model.layer4.2.bn2.weight','model.layer4.2.conv3.weight','model.layer4.2.bn3.weight','model.fc.0.weight','model.fc.2.weight']
                    MAX_VAR_value = [0.0151827205,0.007444239,0.005016849,0.008202254,0.00084967306,0.0030392464,0.0012437532,0.01056604,0.0033067204,0.012968501,0.0009378583,0.0052540815,0.00083003764,0.0011983651,0.0010588814,0.0046438244,0.00090482354,0.000861097,0.0010096661,0.000717722,0.0009663616,0.007723973,0.001254565,0.0014608663,0.00049680984,0.0010352045,0.0007886441,0.007829584,0.0005343012,0.007963212,0.0002775193,0.00086844305,0.00037389874,0.0012430112,0.00048660772,0.009556309,0.00054362934,0.0010163392,0.0004663081,0.0010244737,0.0006850892,0.0037191357,0.00059136556,0.0005846987,0.000502172,0.00084503426,0.000591321,0.0044786306,0.00094655214,0.0010986804,0.00030146254,0.0008508534,0.00054639805,0.003823426,0.00026079596,0.0031182973,0.00023146307,0.0008051725,0.00022794577,0.0015656671,0.00041437388,0.0021250155,0.00024805422,0.0007266049,0.0002251114,0.0010000506,0.0003639597,0.0017320176,0.0003007828,0.00085557054,0.00022277478,0.0007725301,0.0003316295,0.0021655594,0.00032799927,0.0009597795,0.00022118454,0.0007828836,0.00032952012,0.0024006306,0.00038823206,0.0010160859,0.00023230484,0.0014764221,0.0003699709,0.0024253197,0.00056632253,0.0005786456,0.00015761971,0.00045362447,0.00023385315,0.004373357,9.871223e-05,0.0050619836,0.00021673393,0.00048291963,0.00015117564,0.00036030103,0.00022354923,0.006730213,0.00032879718,0.00083312567,0.00012198849,0.0005762829,0.00020090755,0.022952389,0.00040454502,0.002883711]
                    MAX_VAR = dict(zip(MAX_VAR_key,MAX_VAR_value))
                    for name,p in model.named_parameters():
                        if 'weight' in name:
                            if 'bn' in name or 'fc' in name:
                                var = _max_var_lookup(MAX_VAR, name)
                                if var is None:
                                    continue
                                mean = p.data.mean()
                                p.data.clamp_(mean - np.sqrt(var), mean + np.sqrt(var))

            cum_loss += loss.item() * B
            if is_binary:
                cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
            else:
                pred_c = pred.max(1)[1].cpu()
                cum_acc += (pred_c.eq(y_in)).sum().item()
            tot = tot + B
        if verbose:
            print("Epoch %d, loss = %.4f, acc = %.4f" % (epoch, cum_loss / tot, cum_acc / tot))
        if epoch % 5 == 0:
            backdooracc = eval_model(model, testloader_mal, is_binary=is_binary)
            print("backdoor acc = %.4f"%(backdooracc))
            
    return


def eval_model(model, dataloader, is_binary):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i, (x_in, y_in) in enumerate(dataloader):
        B = x_in.size()[0]
        pred = model(x_in)
        if is_binary:
            cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
        else:
            pred_c = pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    return cum_acc / tot
