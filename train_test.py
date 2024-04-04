import torch.utils.data as Data
import conf.config as conf
import argparse
from data.dataLoader import *
from network_structure import VGG16, ResNet50, MobileNetV2
from network_structure.EfficientNet import EfficientNet
from network_structure.ResNet50 import  Bottleneck
import utils
import os
from thop import profile
import torch.nn as nn


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='FCN Training With Pytorch')
parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'FaceScrub', 'ImageNet1000'],
                    type=str, help='CIFAR10, CIFAR100, TinyImageNet, FaceScrubs or ImageNet1000')
parser.add_argument('--dataset_root', default='',
                    help='Dataset root directory path')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--save_folder', default='./models(3.5)/ResNet50/CIFAR100/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():

    args.dataset = 'CIFAR100'  # ImageNet100
    if args.dataset == 'CIFAR100':
        train_data = CIFAR100_train_data
        test_data = CIFAR100_test_data
        cfg = conf.CIFAR100

    elif args.dataset == 'CIFAR10':
        train_data = CIFAR10_train_data
        test_data = CIFAR10_test_data
        cfg = conf.CIFAR10

    elif args.dataset == 'TinyImageNet':
        train_data = TinyImageNet_train_data
        test_data = TinyImageNet_test_data
        cfg = conf.TinyImageNet

    elif args.dataset == 'FaceScrub':
        train_data = FaceScrub_train_data
        test_data = FaceScrub_test_data
        cfg = conf.FaceScrubs

    elif args.dataset == 'ImageNet1000':
        train_data = ImageNet1000_train_data
        test_data = ImageNet1000_test_data
        cfg = conf.ImageNet1000

    else:
        print("dataset doesn't exist!")
        exit(0)


    # ResNet50 and ResNet50_FCN
    cnn = ResNet50.ResNet50(Bottleneck, [3, 4, 6, 3], num_classes=cfg['num_classes'])
    print(cnn)
    criterion = nn.CrossEntropyLoss()
    loss_function = 'Softmax Loss'
    train_loader = Data.DataLoader(dataset=train_data, batch_size=cfg['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=cfg['batch_size'], shuffle=False, num_workers=6, pin_memory=True)

    # start training
    utils.train_test_fcn(cnn, train_loader, test_loader, cfg, criterion, args.save_folder, cfg['num_classes'], loss_function=loss_function)


if __name__ == '__main__':
    train()
