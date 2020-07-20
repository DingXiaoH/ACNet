import torch
from torchvision import datasets, transforms
import numpy as np
from data.dataset_util import InfiniteDataLoader

CIFAR10_PATH = 'cifar10_data'
MNIST_PATH = 'mnist_data'


def load_cuda_data(data_loader, dataset_name):
    if dataset_name == 'imagenet_standard':
        data, label = next(data_loader.dataprovider)
        data = data.cuda()
        label = label.cuda()
    elif dataset_name == 'imagenet_blank':
        data_dict = next(data_loader)
        data = data_dict['data']
        label = data_dict['label']
    else:
        data, label = next(data_loader)
        data = data.cuda()
        label = label.cuda()
    return data, label

class ImageNetBlankGenerator(object):

    def __init__(self, batch_size, img_size):
        assert type(img_size) is int

        self.blank_img = np.ones((batch_size, 3, img_size, img_size), dtype=np.float)
        self.blank_label = np.ones(batch_size, dtype=np.int) * 42
        self.return_dict = {'data': torch.from_numpy(self.blank_img).type(torch.FloatTensor).cuda(),
                            'label': torch.from_numpy(self.blank_label).type(torch.long).cuda()}

    def __next__(self):
        return self.return_dict

def create_dataset(dataset_name, subset, global_batch_size, distributed):
    assert dataset_name in ['cifar10','imagenet_blank',
                            'imagenet_standard', 'mnist']
    assert subset in ['train', 'val']

    if dataset_name == 'imagenet_standard':
        from data.imagenet_data import ImgnetStdTrainData, ImgnetStdValData
        if subset == 'train':
            print('imgnet standard train data')
            return ImgnetStdTrainData(distributed=distributed,
                                      batch_size_per_gpu=global_batch_size // torch.cuda.device_count())
        else:
            print('imgnet standard val data')
            return ImgnetStdValData(batch_size=global_batch_size)

    elif dataset_name == 'imagenet_blank':
        assert not distributed
        return ImageNetBlankGenerator(batch_size=global_batch_size, img_size=224)

    elif dataset_name == 'mnist':
        assert not distributed
        if subset == 'train':
            return InfiniteDataLoader(datasets.MNIST(MNIST_PATH, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))])),
                                      batch_size=global_batch_size, shuffle=True)
        else:
            return InfiniteDataLoader(datasets.MNIST(MNIST_PATH, train=False, transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size=global_batch_size, shuffle=False)

    elif dataset_name == 'cifar10':
        assert not distributed
        if subset == 'train':
            return InfiniteDataLoader(datasets.CIFAR10(CIFAR10_PATH, train=True, download=False,
                               transform=transforms.Compose([
                                   transforms.Pad(padding=(4, 4, 4, 4)),
                                   transforms.RandomCrop(32),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                                batch_size=global_batch_size, shuffle=True)
        else:
            return InfiniteDataLoader(datasets.CIFAR10(CIFAR10_PATH, train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                                batch_size=global_batch_size, shuffle=False)

    else:
        raise ValueError('??')


def num_train_examples_per_epoch(dataset_name):
    if 'imagenet' in dataset_name:
        return 1281167
    elif dataset_name in ['cifar10', 'ch']:
        return 50000
    elif dataset_name == 'mnist':
        return 60000
    else:
        assert False

def num_iters_per_epoch(cfg):
    return num_train_examples_per_epoch(cfg.dataset_name) // cfg.global_batch_size

def num_val_examples(dataset_name):
    if 'imagenet' in dataset_name:
        return 50000
    elif dataset_name in ['cifar10', 'ch', 'mnist']:
        return 10000
    else:
        assert False