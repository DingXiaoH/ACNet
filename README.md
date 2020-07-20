# ACNet

News:
1. Zhang et al. used our ACB in their model [ACFD](https://arxiv.org/abs/2007.00899), which won the **1st place** in IJCAI 2020 iCartoon Face Challenge (Detection Track). Congratulations!
2. Liu et al. extended ACB to EACB (Enhanced Asym Conv Block) in their [MMDM](https://openaccess.thecvf.com/content_CVPRW_2020/html/w31/Liu_MMDM_Multi-Frame_and_Multi-Scale_for_Image_Demoireing_CVPRW_2020_paper.html), which helped them won the **3rd place** in NTIRE 2020 Challenge on Image Demoireing at CVPR 2020. Congratulations!
3. MMDM also won the **4th place** in NTIRE 2020 Challenge on Real Image Denoising at CVPR 2020. Congratulations again!
4. ACNet has been used in several real business products.
5. At ICCV 2019, I was told that ACNet improved the performance of some semantic segmentation tasks by 2%. So glad to hear that!

Update: Updated the whole repo, including **ImageNet** training (with Distributed Data Parallel). The default learning rate schedules were changed to cosine annealing, which yield higher accuracy on ImageNet. Changed the behavior of ACB when k > 3. It used to add 1x3 and 3x1 kernels onto 5x5, but now it uses 1x5 and 5x1.

ICCV 2019 paper: [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf).

Other implementations:
1. [PaddlePaddle](https://github.com/paddlepaddle/paddle) re-implementation for [building ACNet](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/models/resnet_acnet.py) and [converting the weights](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/utils/acnet) has been accepted by PaddlePaddle official repo. Amazing work by @parap1uie-s!
2. Tensorflow2: an easy plugin module (https://github.com/CXYCarson/TF_AcBlock)! Just use it to build your model and call deploy() to convert it into the inference-time structure! Amazing work by @CXYCarson!

This demo will show you how to
1. Build an ACNet with Asymmetric Convolution Block. Just a few lines of code!
2. Train the ACNet together with the regular CNN baseline with the same training configurations.
3. Test the ACNet and the baseline, get the average accuracy.
4. Convert the ACNet into exactly the same structure as the regular counterpart for deployment. Congratulations! The users of your model will be happy because they can enjoy higher accuracy with exactly the same computational burdens as the baseline trained with regular conv layers.

About the environment:
1. We used torch==1.3.0, torchvision==0.4.1, CUDA==10.2, NVIDIA driver version==440.82, tensorboard==1.11.0 on a machine with eight 2080Ti GPUs. 
2. Our method does not rely on any new or deprecated features of any libraries, so there is no need to make an identical environment.
3. If you get any errors regarding tensorboard or tensorflow, you may simply delete the code related to tensorboard or SummaryWriter.

Some results (Top-1 accuracy) reproduced on CIFAR-10 using the codes in this repository (note that we add batch norm for Cifar-quick and VGG baselines):

| Model        | Baseline           | ACNet  |
| ------------- |:-------------:| -----:|
| Cifar-quick   | 86.20 	|  	86.87 |
| VGG      	| 93.99      	|   	94.54 |
| ResNet-56 	| 94.55      	|    	95.06 |
| WRN-16-8 	| 95.89		|    	96.33 |

If it does not work on your specific model and dataset, based on my experience, I would suggest you
1. try different learning rate schedules
2. initialize the trained scaling factor of batch norm (e.g., gamma variable in Tensorflow and bn.weight in PyTorch) in the three branches of every ACB as 1/3. This improves the performance on CIFAR

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

## Example Usage: ResNet-18 on ImageNet with multiple GPUs

1. Enter this directory.

2. Make a soft link to your ImageNet directory, which contains "train" and "val" directories.
```
ln -s YOUR_PATH_TO_IMAGENET imagenet_data
```

3. Set the environment variables. We use 8 GPUs with Distributed Data Parallel. Of course, 4 GPUs work as well.
```
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

3. Train a regular ResNet-18 on ImageNet as baseline.
```
python -m torch.distributed.launch --nproc_per_node=8 acnet/do_acnet.py -a sres18 -b base
```

4. Train a ResNet-18 on ImageNet with Asymmetric Convolution Blocks. The code will automatically convert the trained weights to the original structure and test it.
```
python -m torch.distributed.launch --nproc_per_node=8 acnet/do_acnet.py -a sres18 -b acb
```

5. Check the shape of weights in the converted model.
```
python3 display_hdf5.py acnet_exps/sres18_acb_train/finish_deploy.hdf5
```

## Example Usage: Cifar-quick, VGG, ResNet-56, WRN-16-8 on CIFAR-10 with 1 GPU

1. Enter this directory.

2. Make a soft link to your CIFAR-10 directory. If the dataset is not found in the directory, it will be automatically downloaded.
```
ln -s YOUR_PATH_TO_CIFAR cifar10_data
```

3. Set the environment variables.
```
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
```

4. Train the Cifar-quick baseline and ACNet.
```
python acnet/do_acnet.py -a wrnc16plain -b base
python acnet/do_acnet.py -a wrnc16plain -b acb
```

5. Train the VGG baseline and ACNet.
```
python acnet/do_acnet.py -a vc -b base
python acnet/do_acnet.py -a vc -b acb
```

6. Train the ResNet-56 baseline and ACNet.
```
python acnet/do_acnet.py -a src56 -b base
python acnet/do_acnet.py -a src56 -b acb
```

7. Train the WRN-16-8 baseline and ACNet.
```
python acnet/do_acnet.py -a wrnc16plain -b base
python acnet/do_acnet.py -a wrnc16plain -b acb
```

8. Show the accuracy of all the models.
```
python show_log.py acnet_exps
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

**State-of-the-art** channel pruning (preprint, 2020): [Lossless CNN Channel Pruning via Gradient Resetting and Convolutional Re-parameterization](https://arxiv.org/abs/2007.03260) (https://github.com/DingXiaoH/ResRep)

CNN component (ICCV 2019): [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf) (https://github.com/DingXiaoH/ACNet)

Channel pruning (CVPR 2019): [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html) (https://github.com/DingXiaoH/Centripetal-SGD)

Channel pruning (ICML 2019): [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html) (https://github.com/DingXiaoH/AOFP)

Unstructured pruning (NeurIPS 2019): [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf) (https://github.com/DingXiaoH/GSM-SGD)
