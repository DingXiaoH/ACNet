import torch
from torchvision import datasets, transforms

MNIST_PATH = '/home/dingxiaohan/datasets/torch_mnist/'
CIFAR10_PATH = '/home/dingxiaohan/datasets/cifar-10-batches-py/'
CH_PATH = '/home/dingxiaohan/datasets/torch_ch/'
SVHN_PATH = '/home/dingxiaohan/datasets/torch_svhn/'


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


def create_dataset(dataset_name, subset, batch_size):
    assert dataset_name in ['imagenet', 'cifar10', 'ch', 'svhn', 'mnist']
    assert subset in ['train', 'val']
    if dataset_name == 'imagenet':
        raise ValueError('TODO')

    #   copied from https://github.com/pytorch/examples/blob/master/mnist/main.py
    elif dataset_name == 'mnist':
        if subset == 'train':
            return InfiniteDataLoader(datasets.MNIST(MNIST_PATH, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=True)
        else:
            return InfiniteDataLoader(datasets.MNIST(MNIST_PATH, train=False, transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size=batch_size, shuffle=False)



    elif dataset_name == 'cifar10':
        if subset == 'train':
            return InfiniteDataLoader(datasets.CIFAR10(CIFAR10_PATH, train=True, download=False,
                               transform=transforms.Compose([
                                   transforms.Pad(padding=(4, 4, 4, 4)),
                                   transforms.RandomCrop(32),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                                batch_size=batch_size, shuffle=True)
        else:
            return InfiniteDataLoader(datasets.CIFAR10(CIFAR10_PATH, train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                                batch_size=batch_size, shuffle=False)

    elif dataset_name == 'ch':
        if subset == 'train':
            return InfiniteDataLoader(datasets.CIFAR100(CH_PATH, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.Pad(padding=(4, 4, 4, 4)),
                                   transforms.RandomCrop(32),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                                batch_size=batch_size, shuffle=True)
        else:
            return InfiniteDataLoader(datasets.CIFAR100(CH_PATH, train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                                batch_size=batch_size, shuffle=False)

    else:
        assert False


def num_train_examples_per_epoch(dataset_name):
    if dataset_name == 'imagenet':
        return 1281167
    elif dataset_name == 'mnist':
        return 60000
    elif dataset_name in ['cifar10', 'ch']:
        return 50000
    else:
        assert False