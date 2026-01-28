
import torch
from torch import Tensor, nn
import torchvision
import os
import numpy as np


class Normalize:
    def __init__(self, n_channels, expected_values, variance):
        self.n_channels = n_channels
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, n_channels, expected_values, variance):
        self.n_channels = n_channels
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


class RegressionModel(nn.Module):
    def __init__(self, task, model, init_mask, init_pattern):
        self._EPSILON = 1e-7
        super(RegressionModel, self).__init__()
        self.mask_tanh = nn.Parameter(torch.tensor(init_mask))
        self.pattern_tanh = nn.Parameter(torch.tensor(init_pattern))

        self.classifier = self._get_classifier(model)
        self.normalizer = self._get_normalize(task)
        self.denormalizer = self._get_denormalize(task)

    def forward(self, x):
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
        if self.normalizer:
            pattern = self.normalizer(self.get_raw_pattern())
        x = (1 - mask) * x + mask * pattern
        return self.classifier(x)

    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        return mask / (2 + self._EPSILON) + 0.5

    def get_raw_pattern(self):
        pattern = nn.Tanh()(self.pattern_tanh)
        return pattern / (2 + self._EPSILON) + 0.5

    def _get_classifier(self, model):
        classifier = model
        for param in classifier.parameters():
            param.requires_grad = False
        classifier.eval()
        return classifier.to('cuda')

    def _get_denormalize(self, task):
        if task == 'cifar10':
            denormalizer = Denormalize(3, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif task == 'mnist':
            denormalizer = Denormalize(1, [0.5], [0.5])
        elif task == 'imageNet':
            denormalizer = Denormalize(3,[0.485,0.456,0.406],[0.229,0.224,0.225])
        elif task == 'gtsrb':
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, task):
        if task == 'cifar10':
            normalizer = Normalize(3, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif task == 'mnist':
            normalizer = Normalize(1, [0.5], [0.5])
        elif task == 'imageNet':
            normalizer = Denormalize(3,[0.485,0.456,0.406],[0.229,0.224,0.225])
        elif task == 'gtsrb':
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer


class Recorder:
    def __init__(self,target_label,task):
        super().__init__()

        # Best optimization results
        self.mask_best = None
        self.pattern_best = None
        self.reg_best = float('inf')
        self.target_label = target_label

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.reg_best

        # Cost
        self.cost = 1e-3
        self.cost_multiplier_up = 2
        self.cost_multiplier_down = 2 ** 1.5

        self.task = task

    def reset_state(self):
        self.cost = 1e-3
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        print("Initialize cost to {:f}".format(self.cost))

    def save_result_to_dir(self):
        result_dir = '%s/u_t_%s'%(self.task,self.task)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_dir = os.path.join(result_dir, str(self.target_label))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        pattern_best = self.pattern_best
        mask_best = self.mask_best
        trigger = pattern_best * mask_best

        path_mask = os.path.join(result_dir, 'mask.png')
        path_pattern = os.path.join(result_dir, 'pattern.png')
        path_trigger = os.path.join(result_dir, 'trigger.png')

        torchvision.utils.save_image(mask_best, path_mask, normalize=True)
        torchvision.utils.save_image(pattern_best, path_pattern, normalize=True)
        torchvision.utils.save_image(trigger, path_trigger, normalize=True)


class UniversalTrigger:
    def __init__(self):
        self.universal_trigger_dict = {}


def train(task, model, target_label, init_mask, init_pattern, test_loader):
    # Build regression model
    regression_model = RegressionModel(task, model, init_mask, init_pattern).to('cuda')

    # Set optimizer
    optimizerR = torch.optim.Adam(regression_model.parameters(), lr=1e-1, betas=(0.5, 0.9))

    # Set recorder (for recording best result)
    recorder = Recorder(target_label,task)

    for epoch in range(25):
        early_stop = train_step(regression_model, optimizerR, test_loader, recorder, epoch, target_label)
        if early_stop:
            break

    # Save result to dir
    recorder.save_result_to_dir()

    return recorder


def train_step(regression_model, optimizerR, dataloader, recorder, epoch, target_label,early_stop=True):
    print("Epoch {} - Label: {}".format(epoch, target_label))
    # Set losses
    cross_entropy = nn.CrossEntropyLoss()
    total_pred = 0
    true_pred = 0

    # Record loss for all mini-batches
    loss_ce_list = []
    loss_reg_list = []
    loss_list = []
    loss_acc_list = []

    # Set inner early stop flag
    inner_early_stop_flag = False
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # Forwarding and update model
        optimizerR.zero_grad()

        inputs = inputs.to('cuda')
        sample_num = inputs.shape[0]
        total_pred += sample_num
        target_labels = torch.ones((sample_num), dtype=torch.int64).to('cuda') * target_label
        predictions = regression_model(inputs)

        loss_ce = cross_entropy(predictions, target_labels)
        loss_reg = torch.norm(regression_model.get_raw_mask(), 2)
        total_loss = loss_ce + recorder.cost * loss_reg
        total_loss.backward()
        optimizerR.step()

        # Record minibatch information to list
        minibatch_accuracy = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() * 100. / sample_num
        loss_ce_list.append(loss_ce.detach())
        loss_reg_list.append(loss_reg.detach())
        loss_list.append(total_loss.detach())
        loss_acc_list.append(minibatch_accuracy)

        true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()

    loss_ce_list = torch.stack(loss_ce_list)
    loss_reg_list = torch.stack(loss_reg_list)
    loss_list = torch.stack(loss_list)
    loss_acc_list = torch.stack(loss_acc_list)

    avg_loss_ce = torch.mean(loss_ce_list)
    avg_loss_reg = torch.mean(loss_reg_list)
    avg_loss = torch.mean(loss_list)
    avg_loss_acc = torch.mean(loss_acc_list)

    # Check to save best mask or not
    if avg_loss_acc >= 99. and avg_loss_reg < recorder.reg_best:
        recorder.mask_best = regression_model.get_raw_mask().detach()
        recorder.pattern_best = regression_model.get_raw_pattern().detach()
        recorder.reg_best = avg_loss_reg
        recorder.save_result_to_dir()
        print(" Updated !!!")

    # Show information
    print('  Result: Accuracy: {:.3f} | Cross Entropy Loss: {:.6f} | Reg Loss: {:.6f} | Reg best: {:.6f}'.format(
        true_pred * 100. / total_pred,
        avg_loss_ce,
        avg_loss_reg,
        recorder.reg_best))

    # Check early stop
    if early_stop:
        if recorder.reg_best < float('inf'):
            if recorder.reg_best >= 99. * recorder.early_stop_reg_best:
                recorder.early_stop_counter += 1
            else:
                recorder.early_stop_counter = 0

        recorder.early_stop_reg_best = min(recorder.early_stop_reg_best, recorder.reg_best)

        if (
                recorder.cost_down_flag and recorder.cost_up_flag and recorder.early_stop_counter >= 25):
            print('Early_stop !!!')
            inner_early_stop_flag = True

    if not inner_early_stop_flag:
        # Check cost modification
        if recorder.cost == 0 and avg_loss_acc >= 99.:
            recorder.cost_set_counter += 1
            if recorder.cost_set_counter >= 5:
                recorder.reset_state()
        else:
            recorder.cost_set_counter = 0

        if avg_loss_acc >= 99.:
            recorder.cost_up_counter += 1
            recorder.cost_down_counter = 0
        else:
            recorder.cost_up_counter = 0
            recorder.cost_down_counter += 1

        if recorder.cost_up_counter >= 5:
            recorder.cost_up_counter = 0
            print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
            recorder.cost *= recorder.cost_multiplier_up
            recorder.cost_up_flag = True

        elif recorder.cost_down_counter >= 5:
            recorder.cost_down_counter = 0
            print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
            recorder.cost /= recorder.cost_multiplier_down
            recorder.cost_down_flag = True

        # Save the final version
        if recorder.mask_best is None:
            recorder.mask_best = regression_model.get_raw_mask().detach()
            recorder.pattern_best = regression_model.get_raw_pattern().detach()

    return inner_early_stop_flag
