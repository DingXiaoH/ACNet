import math
import os
from typing import Tuple, List, Dict
import torch
import sys
import json
import h5py
import numpy as np
import time

def init_as_tensorflow(model):
    for k, v in model.named_parameters():
        if v.dim() in [2, 4]:
            torch.nn.init.xavier_uniform_(v)
            print('init {} as xavier_uniform'.format(k))
        if 'bias' in k and 'bn' not in k.lower():
            torch.nn.init.zeros_(v)
            print('init {} as zero'.format(k))


def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor

def torchvision_style_normalize(input):
    input_arr = input.cpu().numpy()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for kk in range(input_arr.shape[0]):
        im = input_arr[kk, :, :, :]
        im = im[::-1, :, :]
        im /= 255.
        im = im.transpose((1, 2, 0))
        im -= mean
        im /= std
        im = im.transpose((2, 0, 1))
        input_arr[kk, :, :, :] = im
    input_tensor = torch.from_numpy(input_arr).type(input.dtype).to(input.device)
    return input_tensor

torchvision_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
torchvision_std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

def efficient_torchvision_style_normalize(input):
    input_arr = input.cpu().numpy()
    input_arr /= 255.

    input_arr = np.array(input_arr[:, ::-1, :, :])
    # input_arr = input_arr.transpose((0, 2, 3, 1))
    input_arr -= torchvision_mean
    input_arr /= torchvision_std

    # for kk in range(input_arr.shape[0]):
    #     im = input_arr[kk, :, :, :]
    #     im = im[::-1, :, :]
    #     im = im.transpose((1, 2, 0))    # from CHW to HWC
    #     im -= mean
    #     im /= std
    #     im = im.transpose((2, 0, 1))    # from HWC to CHW
    #     input_arr[kk, :, :, :] = im
    input_tensor = torch.from_numpy(input_arr).type(input.dtype).to(input.device)
    return input_tensor


def cur_time():
    return time.strftime('%Y,%b,%d,%X')

def log_important(message, log_file):
    print(message, cur_time())
    with open(log_file, 'a') as f:
        print(message, cur_time(), file=f)

def extract_deps_from_weights_file(file_path):
    weight_dic = read_hdf5(file_path)
    if 'deps' in weight_dic:
        return weight_dic['deps']
    else:
        return None

def representsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def read_hdf5(file_path):
    result = {}
    with h5py.File(file_path, 'r') as f:
        for k in f.keys():
            value = np.asarray(f[k])
            if representsInt(k):
                result[int(k)] = value
            else:
                result[str(k).replace('+','/')] = value
    print('read {} arrays from {}'.format(len(result), file_path))
    f.close()
    return result

def save_hdf5(numpy_dict, file_path):
    with h5py.File(file_path, 'w') as f:
        for k,v in numpy_dict.items():
            f.create_dataset(str(k).replace('/','+'), data=v)
    print('saved {} arrays to {}'.format(len(numpy_dict), file_path))
    f.close()


def start_exp():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--try_arg", type=str, default='')
    args = parser.parse_args()
    try_arg = args.try_arg
    print('the try_arg is ', try_arg)
    print('we have {} torch devices'.format(torch.cuda.device_count()),
          'the allocated GPU memory is {}'.format(torch.cuda.memory_allocated()))
    return try_arg


def start_exp_argv():
    import sys
    try_arg = sys.argv[1]
    assert '--try' not in try_arg
    print('the try_arg is ', try_arg)
    print('we have {} torch devices'.format(torch.cuda.device_count()),
          'the allocated GPU memory is {}'.format(torch.cuda.memory_allocated()))
    return try_arg

def start_exp_model_and_argv():
    import sys
    model_name = sys.argv[1]
    try_arg = sys.argv[2]
    assert '--try' not in try_arg
    print('the try_arg is ', try_arg)
    print('we have {} torch devices'.format(torch.cuda.device_count()),
          'the allocated GPU memory is {}'.format(torch.cuda.memory_allocated()))
    if len(sys.argv) >= 4:
        assert len(sys.argv) == 4
        assert sys.argv[3] == 'continue'
        auto_continue = True
    else:
        auto_continue = False
    return model_name, try_arg, auto_continue


def start_exp_model_and_argv_with_local_rank():
    import sys
    model_name = sys.argv[2]
    try_arg = sys.argv[3]
    assert '--try' not in try_arg
    print('the try_arg is ', try_arg)
    print('we have {} torch devices'.format(torch.cuda.device_count()),
          'the allocated GPU memory is {}'.format(torch.cuda.memory_allocated()))
    if len(sys.argv) >= 5:
        assert len(sys.argv) == 5
        assert sys.argv[4] == 'continue'
        auto_continue = True
    else:
        auto_continue = False
    return model_name, try_arg, auto_continue


def torch_accuracy(output, target, topk=(1,)) -> List[torch.Tensor]:
    '''
    param output, target: should be torch Variable
    '''
    # assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
    # assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
    # print(type(output))

    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans

class AvgMeter(object):
    '''
    Computing mean
    '''
    name = 'No name'

    def __init__(self, name='No name', fmt = ':.2f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0

    def update(self, mean_var, count=1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num

    def __str__(self):
        print_str = self.name + '-{' + self.fmt + '}'
        return print_str.format(self.mean)

def save_args(args, save_dir = None):
    if save_dir == None:
        param_path = os.path.join(args.resume, "params.json")
    else:
        param_path = os.path.join(save_dir, 'params.json')

    #logger.info("[*] MODEL dir: %s" % args.resume)
    #logger.info("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)


def mkdir(path):
    if not os.path.exists(path):
        print('creating dir {}'.format(path))
        os.mkdir(path)

# def save_checkpoint(cur_iters, net, optimizer, lr_scheduler, file_name):
#     checkpoint = {'cur_iters': cur_iters,
#                   'state_dict': net.state_dict(),
#                   'optimizer_state_dict': optimizer.state_dict(),
#                   'lr_scheduler_state_dict':lr_scheduler.state_dict()}
#     if os.path.exists(file_name):
#         print('Overwriting {}'.format(file_name))
#     torch.save(checkpoint, file_name)
#     link_name = os.path.join('/', *file_name.split(os.path.sep)[:-1], 'last.checkpoint')
#     #print(link_name)
#     make_symlink(source = file_name, link_name=link_name)

def load_checkpoint(file_name, net = None, optimizer = None, lr_scheduler = None):
    if os.path.isfile(file_name):
        print("=> loading checkpoint '{}'".format(file_name))
        check_point = torch.load(file_name)
        if net is not None:
            print('Loading network state dict')
            net.load_state_dict(check_point['state_dict'])
        if optimizer is not None:
            print('Loading optimizer state dict')
            optimizer.load_state_dict(check_point['optimizer_state_dict'])
        if lr_scheduler is not None:
            print('Loading lr_scheduler state dict')
            lr_scheduler.load_state_dict(check_point['lr_scheduler_state_dict'])

        return check_point['cur_iters']
    else:
        print("=> no checkpoint found at '{}'".format(file_name))


def make_symlink(source, link_name):
    '''
    Note: overwriting enabled!
    '''
    if os.path.exists(link_name):
        #print("Link name already exist! Removing '{}' and overwriting".format(link_name))
        os.remove(link_name)
    if os.path.exists(source):
        os.symlink(source, link_name)
        return
    else:
        print('Source path not exists')
    #print('SymLink Wrong!')

def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)

def format_metric_dict_to_line(metric_dict):
    msg = ''
    for key, value in metric_dict.items():
        msg += '{}={:.5f},'.format(key, value)
    return msg
