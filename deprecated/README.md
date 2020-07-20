# ACNet

**These are deprecated.**

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

