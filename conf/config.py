# config.py
import os.path


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
HOME = os.path.expanduser("~")
CMN_scale = 1.0

# different DATA set configs
CIFAR10 = {
    'num_classes': 10,
    'feature_sizes': 256,
    'max_epoch': 200,
    'LR': 1e-1,
    'lr_steps': (0, 80, 120, 150, 180),
    'batch_size': 128,
    'name': 'CIFAR10'
}
CIFAR100 = {
    'num_classes': 100,
    'feature_sizes': 512,
    'max_epoch':200,
    'LR':1e-1,
    'lr_steps':(0,80,120,150,180),
    'batch_size': 128,
    'name': 'CIFAR100'
}
TinyImageNet = {
    'num_classes': 200,
    'feature_sizes': 512,
    'max_epoch':200,
    'LR':1e-1,
    'lr_steps':(0,80,120,150,180),
    'batch_size': 128,
    'name': 'TinyImageNet'
}
FaceScrubs = {
    'num_classes': 100,
    'feature_sizes': 512,
    'max_epoch':200,
    'LR':1e-1,
    'lr_steps':(0, 80, 120, 150, 180),
    'batch_size': 128,
    'name': 'FaceScrubs'
}
# ImageNet100
ImageNet100 = {
    'num_classes':100,
    'max_epoch':150,
    'LR':1e-1,
    'lr_steps':(0,30,60,90,120),
    'feature_size':512,
    'batch_size':128,                # efficientNet:96
    'name':'Imagenet100'
}


ImageNet1000 = {
    'num_classes': 1000,
    'feature_sizes': 2048, #Varies according to the number of classes, the corresponding relationship between them has been given in our paper.
    'max_epoch': 100,
    'LR': 1e-1,   # Set according to the specific situation
    'lr_steps': (0, 30, 60, 90),   #Set according to the specific situation
    'batch_size': 128,
    'name': 'ImageNet1000'
}
