import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import utils_avatar as ua
import os
from datetime import datetime
import json
import argparse
import time
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/gtsrb/imageNet).')
parser.add_argument('--arch', type=str, default='cnn', help='Model arch for cifar10 (cnn|resnet18).')
parser.add_argument('--to_file', action='store_true', help='If set, dump the result to file.')
parser.add_argument('--data-root', type=str, default='./data', help='Dataset root (already prepared to avoid download).')

def train_model(task, model, dataloader,epoch_num, is_binary, verbose=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
            cum_loss += loss.item() * B
            if is_binary:
                cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
            else:
                pred_c = pred.max(1)[1].cpu()
                cum_acc += (pred_c.eq(y_in)).sum().item()
            tot = tot + B
        if verbose:
            print("Epoch %d, loss = %.4f, acc = %.4f" % (epoch, cum_loss / tot, cum_acc / tot))
            
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


if __name__ == '__main__':
    args = parser.parse_args()

    GPU = True
    TARGET_PROP = 1.0
    np.random.seed(0)
    torch.manual_seed(0)
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, need_pad, Model, input_size, class_num = ua.load_dataset_setting(
        args.task, data_root=args.data_root, arch=args.arch, download=False)
    tot_num = len(trainset)
    target_indices = np.random.choice(tot_num, int(tot_num * TARGET_PROP))
    print("Data indices owned by the attacker:", target_indices)
    target_set = torch.utils.data.Subset(trainset, target_indices)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    SAVE_PREFIX = os.path.join(script_dir, f"{args.task}" if args.arch == 'cnn' else f"{args.task}_{args.arch}")
    if not os.path.isdir(SAVE_PREFIX):
        os.mkdir(SAVE_PREFIX)
    if not os.path.isdir(SAVE_PREFIX + '/models'):
        os.mkdir(SAVE_PREFIX + '/models')

    for i in range(1):
        t1 = time.time()
        model = Model(gpu=GPU)
        cleantrainloader = torch.utils.data.DataLoader(target_set, batch_size=BATCH_SIZE, shuffle=True)
        cleantestloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

        train_model(args.task, model, cleantrainloader, epoch_num=N_EPOCH, is_binary=is_binary,verbose=True)
        cleanacc = eval_model(model, cleantestloader, is_binary=is_binary)
        torch.save(model.state_dict(),'%s/models/clean_%i.model'%(SAVE_PREFIX,i))
        print("Clean model %d : Acc %.4f @ %s" % (i, cleanacc, datetime.now()))
        t2 = time.time()
        print('Generate clean model cost %.3fs.'%(t2-t1))

        output_path = SAVE_PREFIX +'/clean_output.txt'

        if args.to_file:
            with open(output_path, 'a+') as f:
                f.write("Clean model %d : Acc %.4f @ %s\n" % (i, cleanacc, datetime.now()))
