# ACNet
Note: there are some bugs. I am checking. (2019/11/05)

News:
1. ACNet has been used in real business products.
2. At ICCV 2019, I was told that ACNet improved the performance of some semantic segmentation tasks by 2%. So glad to hear that!

This repository contains the codes for the following ICCV-2019 paper 

[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf).

The codes are based on PyTorch 1.1.

The experiments reported in the paper were performed using Tensorflow. However, the backbone of the codes was refactored from the official Tensorflow benchmark (https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks), which was designed in the pursuit of extreme speed, not readability.

Citation:

	@InProceedings{Ding_2019_ICCV,
	author = {Ding, Xiaohan and Guo, Yuchen and Ding, Guiguang and Han, Jungong},
	title = {ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks},
	booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
	month = {October},
	year = {2019}
	}

## Introduction

As designing appropriate Convolutional Neural Network (CNN) architecture in the context of a given application usually involves heavy human works or numerous GPU hours, the research community is soliciting the architecture-neutral CNN structures, which can be easily plugged into multiple mature architectures to improve the performance on our real-world applications. We propose Asymmetric Convolution Block (ACB), an architecture-neutral structure as a CNN building block, which uses 1D asymmetric convolutions to strengthen the square convolution kernels. For an off-the-shelf architecture, we replace the standard square-kernel convolutional layers with ACBs to construct an Asymmetric Convolutional Network (ACNet), which can be trained to reach a higher level of accuracy. After training, we equivalently convert the ACNet into the same original architecture, thus requiring no extra computations anymore. We have observed that ACNet can improve the performance of various models on CIFAR and ImageNet by a clear margin. Through further experiments, we attribute the effectiveness of ACB to its capability of enhancing the model's robustness to rotational distortions and strengthening the central skeleton parts of square convolution kernels.

## Example Usage
  
This repo holds the example codes for training ResNet-56 and WRN-16-8 on CIFAR-10.

1. Install PyTorch 1.1

2. Train a ResNet-56 on CIFAR-10 without Asymmetric Convolution Blocks as baseline
```
python acnet_rc56.py --try_arg=normal_lrs1
```
3. Train a ResNet-56 on CIFAR-10 with Asymmetric Convolution Blocks
```
python acnet_rc56.py --try_arg=acnet_lrs1
```
4. Build a ResNet-56 with the same structure as the baseline model, then convert the weights of the ACNet counterpart to initialize it.

TODO: I will implement these codes when I get my GPUs busy on experiments for my next paper and start to play with PyTorch. Pull requests are welcomed.


On WideResNet (WRN-16-8):
```
python acnet_wrnc16.py --try_arg=normal_lrs2
python acnet_wrnc16.py --try_arg=acnet_lrs2
```


## TODOs. 
1. Design experiments to show that an ACB is not equivalent to a regular conv layer by showing the difference between the outputs from a normal conv layer and an ACB with the same inputs.
2. Support more networks.


## Contact
dxh17@mails.tsinghua.edu.cn

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en
