import torchvision
import torchvision.transforms as transforms


# You can find these data sets online
CIFAR10_Train_ROOT = r'F:\dataset\CIFAR10\CIFAR10\train'
CIFAR10_Test_ROOT = r'F:\dataset\CIFAR10\CIFAR10\test'

CIFAR100_Train_ROOT = r'F:\dataset\CIFAR100\CIFAR100\train'
CIFAR100_Test_ROOT = r'F:\dataset\CIFAR100\CIFAR100\test'

FaceScrub_Train_ROOT = r'F:\dataset\FaceScrubs\train_flip_64'
FaceScrub_Test_ROOT = r'F:\dataset\FaceScrubs\test_flip_64'

TinyImageNet_Train_ROOT = r'F:\dataset\TinyImageNet\train'
TinyImageNet_Test_ROOT = r'F:\dataset\TinyImageNet\test'

ImageNet100_Train_ROOT = r'F:\dataset\TinyImageNet\train'
ImageNet100_Test_ROOT = r'F:\dataset\TinyImageNet\test'

ImageNet1000_Train_ROOT = r'F:\dataset\TinyImageNet\train'
ImageNet1000_Test_ROOT = r'F:\dataset\TinyImageNet\test'


#CIFAR10 data set
CIFAR10_train_data = torchvision.datasets.ImageFolder(CIFAR10_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)
CIFAR10_test_data = torchvision.datasets.ImageFolder(CIFAR10_Test_ROOT,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)

# CIFAR100 data set
CIFAR100_train_data = torchvision.datasets.ImageFolder(CIFAR100_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
)
CIFAR100_test_data = torchvision.datasets.ImageFolder(CIFAR100_Test_ROOT,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
)

# TinyImageNet data set
TinyImageNet_train_data = torchvision.datasets.ImageFolder(TinyImageNet_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)
TinyImageNet_test_data = torchvision.datasets.ImageFolder(TinyImageNet_Test_ROOT,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)

# FaceScrubs data set
FaceScrub_train_data = torchvision.datasets.ImageFolder(FaceScrub_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)
FaceScrub_test_data = torchvision.datasets.ImageFolder(FaceScrub_Test_ROOT,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)

# ImageNet1000 data set
ImageNet1000_train_data = torchvision.datasets.ImageFolder(ImageNet1000_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)
ImageNet1000_test_data = torchvision.datasets.ImageFolder(ImageNet1000_Test_ROOT,
    transform=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)
