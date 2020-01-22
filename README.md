# ACNet

News:
1. ACNet has been used in real business products.
2. At ICCV 2019, I was told that ACNet improved the performance of some semantic segmentation tasks by 2%. So glad to hear that!
3. Deployed in more business products.

Update: weights of ACNet on ImageNet will be released in several days.

Update: [PaddlePaddle](https://github.com/paddlepaddle/paddle) re-implementation for [building ACNet](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/resnet_acnet.py) and [converting the weights](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/utils/acnet) has been accepted by PaddlePaddle official repo. Amazing work by @parap1uie-s!

This repository contains the codes for the following ICCV-2019 paper 

[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf).

This demo will show you how to
1. Build an ACNet with Asymmetric Convolution Block. Just a few lines of code!
2. Train the ACNet together with the regular CNN baseline with the same training configurations.
3. Test the ACNet and the baseline, get the average accuracy.
4. Convert the ACNet into exactly the same structure as the regular counterpart for deployment. Congratulations! The users of your model will be happy because they can enjoy higher accuracy with exactly the same computational burdens as the baseline trained with regular conv layers.

Some results (Top-1 accuracy) reproduced on CIFAR-10 using the codes in this repository (note that we add batch norm for Cifar-quick and VGG baselines):

| Model        | Baseline           | ACNet  |
| ------------- |:-------------:| -----:|
| Cifar-quick   | 86.249 	|  	87.102 |
| VGG      	| 94.250      	|   	94.862 |
| ResNet-56 	| 94.573      	|    	94.864 |
| WRN-16-8 	| 95.582	|    	95.920 |

If it does not work on your specific model and dataset, based on my experience, I would suggest you
1. try different learning rate schedules
2. initialize the trained scaling factor of batch norm (e.g., gamma variable in Tensorflow and bn.weight in PyTorch) in the three branches of every ACB as 1/3.

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

1. Install PyTorch 1.1. Clone this repo and enter the directory. Modify PYTHONPATH or you will get an ImportError.
```
export PYTHONPATH='WHERE_YOU_CLONED_THIS_REPO'
```

2. Modify 'CIFAR10_PATH' in dataset.py to the directory of your CIFAR-10 dataset. If the dataset is not found in that directory, it will be automatically downloaded. 

3. Train a Cifar-quick on CIFAR-10 without Asymmetric Convolution Blocks as baseline. (We use learning rate warmup and weight decay on bias parameters. They are not necessities but just preferences. Here 'lrs5' is a pre-defined learning rate schedule.) The model will be evaluated every two epochs.
```
python acnet/acnet_cfqkbnc.py --try_arg=normal_lrs5_warmup_bias
```

4. Train a Cifar-quick on CIFAR-10 with Asymmetric Convolution Blocks. The trained weights will be saved to acnet_exps/cfqkbnc_acnet_lrs5_warmup_bias_train/finish.hdf5. Note that Cifar-quick uses 5x5 convs, and we add 1x3 and 3x1 onto 5x5 kernels. Of course, 1x5 and 5x1 convs may work better.
```
python acnet/acnet_cfqkbnc.py --try_arg=acnet_lrs5_warmup_bias
```

4. Check the average accuracy of the two models in their last ten evaluations. You will see the gap.
```
python show_log.py
```

5. Build a Cifar-quick with the same structure as the baseline model, then convert the weights of the ACNet counterpart via BN fusion and branch fusion to initialize it. Test before and after the conversion. You will see identical results.
```
python acnet/acnet_test.py cfqkbnc acnet_exps/cfqkbnc_acnet_lrs5_warmup_bias_train/finish.hdf5
```

6. Check the name and shape of the converted weights.
```
python display_hdf5.py acnet_exps/cfqkbnc_acnet_lrs5_warmup_bias_train/finish_deploy.hdf5
```

Other models:

VGG is deeper, so we train it for longer:
```
python acnet/acnet_vc.py --try_arg=acnet_lrs3_warmup_bias
```
ResNet-56:
```
python acnet/acnet_rc56.py --try_arg=acnet_lrs3_warmup_bias
```
WRN-16-8, we slightly lengthen the learning rate schedule recommended in the WRN paper:
```
python acnet/acnet_wrnc16.py --try_arg=acnet_lrs6_warmup_bias
```

## TODOs. 
1. Support more networks.
2. Release a PyTorch module so that you can use Asymmetric Convolution Block just like the following example. Pull requests are welcomed.
```
from acnet import AsymConvBlock, acnet_fuse_and_load, acnet_switch_to_deploy

# build model, replace regular Conv2d with AsymConvBlock
class YourNet(nn.module):
    ...
    self.conv1 = AsymConvBlock(in_channels=..., out_channels=..., ...)
    self.conv2 = AsymConvBlock(in_channels=..., out_channels=..., ...)

# train
model = YourNet(...)
train(model)
model.save_checkpoint(SAVE_PATH)	# use just the same PyTorch functions

# deploy
acnet_switch_to_deploy()
deploy_model = YourNet(...)			# here deploy_model should be of the same structure as baseline
acnet_fuse_and_load(deploy_model, SAVE_PATH)	# use the converted weights to initliaze it
test(model)
```


## Contact
dxh17@mails.tsinghua.edu.cn

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

My open-sourced papers and repos:

CNN component (ICCV 2019): [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf) (https://github.com/DingXiaoH/ACNet)

Channel pruning (CVPR 2019): [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html) (https://github.com/DingXiaoH/Centripetal-SGD)

Channel pruning (ICML 2019): [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html) (https://github.com/DingXiaoH/AOFP)

Unstructured pruning (NeurIPS 2019): [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf) (https://github.com/DingXiaoH/GSM-SGD)
