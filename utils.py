from datetime import datetime
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import mpl_toolkits.mplot3d as p3d
import math
import os
import torch.optim as optim
from scipy.special import binom
import scipy.io as io
import pickle
from tqdm import tqdm
# from matplotlib import pyplot as plt
from conf.config import CMN_scale,device_ids


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def get_acc_top5(output, label):
    total = output.shape[0]
    _, pred = output.topk(5, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    correct_k = correct[:5].view(-1).float().sum(0)
    return correct_k / total


    
def train_test_fcn(net, train_data, valid_data, cfg, criterion, save_folder, classes_num, loss_function):
    LR = cfg['LR']
    if torch.cuda.is_available():
        # net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()

    prev_time = datetime.now()
    loss_train = []
    loss_val = []
    acc_train = []
    acc_val = []
    for epoch in range(cfg['max_epoch']):
        if epoch in cfg['lr_steps']:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)# The weight_decay of ImageNet is 1e-4
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in tqdm(train_data):
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
                output = net(im)
                loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_acc += get_acc(output, label)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()
                with torch.no_grad():
                    output = net(im)
                    loss = criterion(output, label)
                    valid_loss += loss.data
                    valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d TrainLoss: %f TrainAcc: %f ValidLoss: %f ValidAcc: %f LR: %f"
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data), optimizer.state_dict()['param_groups'][0]["lr"],))
            loss_train.append(train_loss / len(train_data))
            loss_val.append(valid_loss / len(valid_data))
            acc_train.append(train_acc / len(train_data))
            acc_val.append(valid_acc / len(valid_data))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open(save_folder+f'channelNorm_{CMN_scale}.txt', 'a+')
        if epoch == 0:
            f.write(f'CMN_scale: {CMN_scale}, batchsize:{cfg["batch_size"]}, '
                    f'lr:{cfg["lr_steps"]},GPU:{os.environ["CUDA_VISIBLE_DEVICES"]} '
                    f'begin_time: {cur_time}'.center(130, '=') + '\n')
        f.write(epoch_str + time_str + '\n')
        if epoch + 1 == cfg['max_epoch']:
            f.write(f'end_time: {cur_time}'.center(100, '=') + '\n')
        f.close()
        if (epoch + 1) % 10 == 0:
            torch.save(net, save_folder
            + f'{cfg["name"]}_e={epoch + 1}_channelNorm={CMN_scale}.pth')
