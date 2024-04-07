# CMN
Channel Modulus Normalization for CNN Image Classification

# Requirements
* Python >= 3.7
* Pytorch >= 1.4.0
* torchvision

## Data(common datasets)
* CIFAR10
* CIFAR100
* Tiny ImageNet
* FaceScrubs
* ImageNet1000
```python
python data\dataLoader.py contains pre-processing before training on each data set
```

## Config
```
python conf\config.py contains the configuration of each data set during the training process
```
## Net(common networks)
* VGG16 
* ResNet50 
* DenseNet121 
* MobileNetV2 
* Efficient-b0
```
python network_struture includes the above network
```

## Train and Test
```
python train_test\train_test.py
```
