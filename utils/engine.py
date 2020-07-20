# encoding: utf-8

import os
import os.path as osp
import time
import argparse

import torch
import torch.distributed as dist

from collections import OrderedDict
from utils.pyt_utils import parse_torch_devices, extant_file, link_file, ensure_dir
from utils.logger import get_logger
from utils.checkpoint import load_model
from utils.misc import read_hdf5, save_hdf5


class State(object):
    def __init__(self):
        self.iteration = 0
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def register(self, **kwargs):
        for k, v in kwargs.items():
            assert k in ['iteration', 'model', 'optimizer', 'scheduler']
            setattr(self, k, v)


class Engine(object):
    def __init__(self, local_rank, for_val_only=False, base_config=None):
        self.version = 0.01
        self.state = State()
        self.devices = None
        self.distributed = False
        self.logger = None
        self.base_config = base_config

        # if custom_parser is None:
        #     self.parser = argparse.ArgumentParser()
        # else:
        #     assert isinstance(custom_parser, argparse.ArgumentParser)
        #     self.parser = custom_parser

        # self.inject_default_parser()    #TODO
        # self.args = self.parser.parse_args()

        # self.continue_state_object = self.args.continue_fpath

        num_gpus = torch.cuda.device_count()
        self.local_rank = local_rank
        if num_gpus > 1 and not for_val_only:
            self.distributed = True
            #   single machine
            if 'WORLD_SIZE' not in os.environ:
                assert 'RANK' not in os.environ
                self.world_size = 1
                self.world_rank = 0
                os.environ['WORLD_SIZE'] = self.world_size
                os.environ['RANK'] = self.world_rank
            #   multi machine
            else:
                self.world_size = int(os.environ['WORLD_SIZE'])
                self.world_rank = int(os.environ['RANK'])

            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            dist.barrier()
            self.devices = [i for i in range(self.world_size)]
        else:
            self.world_size = 1
            self.world_rank = 1
            # self.devices = parse_torch_devices(self.args.devices)TODO
            self.devices = [0]


    def setup_log(self, name='train', log_dir=None, file_name=None):
        if self.local_rank == 0:
            self.logger = get_logger(
                name, log_dir, self.local_rank, filename=file_name)
        else:
            self.logger = None

        # if not self.logger:
        #     self.logger = get_logger(
        #         name, log_dir, self.local_rank, filename=file_name)
        # else:
        #     self.logger.warning('already exists logger')
        return self.logger

    # def inject_default_parser(self):
    #     p = self.parser
    #     p.add_argument(
    #         '-d', '--devices', default='0',
    #         help='set data parallel training')
    #     p.add_argument(
    #         '-c', '--continue', type=extant_file, metavar="FILE",
    #         dest="continue_fpath",
    #         help='continue from one certain checkpoint')
    #     p.add_argument(
    #         '--local_rank', default=0, type=int,
    #         help='process rank on node')

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def update_iteration(self, iteration):
        self.state.iteration = iteration

    def save_checkpoint(self, path):
        if self.local_rank > 0:
            return

        # self.logger.info("Saving checkpoint to file {}".format(path))
        t_start = time.time()

        state_dict = {}
        new_state_dict = OrderedDict()

        for k, v in self.state.model.state_dict().items():
            key = k
            if k.split('.')[0] == 'module':
                key = k[7:]
            new_state_dict[key] = v
        state_dict['model'] = new_state_dict

        if self.state.optimizer:
            state_dict['optimizer'] = self.state.optimizer.state_dict()
        if self.state.scheduler:
            state_dict['scheduler'] = self.state.scheduler.state_dict()
        if self.state.iteration:
            state_dict['iteration'] = self.state.iteration

        t_io_begin = time.time()
        try:
            torch.save(state_dict, path)
        except:
            print('save {} failed, continue training'.format(path))
        t_end = time.time()

        del state_dict
        del new_state_dict

        # self.logger.info(
        #     "Save checkpoint to file {}, "
        #     "Time usage:\n\tprepare snapshot: {}, IO: {}".format(
        #         path, t_io_begin - t_start, t_end - t_io_begin))

    def load_checkpoint(self, weights, is_restore=False):

        t_start = time.time()

        loaded = torch.load(weights, map_location=torch.device("cpu"))

        t_io_end = time.time()
        if "model" not in loaded:
            loaded = dict(model=loaded)

        self.state.model = load_model(
            self.state.model, loaded['model'], self.logger)     # TODO is_restore?

        if "optimizer" in loaded:
            self.state.optimizer.load_state_dict(loaded['optimizer'])
        if "iteration" in loaded:
            self.state.iteration = loaded['iteration']
        if "scheduler" in loaded:
            self.state.scheduler.load_state_dict(loaded["scheduler"])
        del loaded

        t_end = time.time()
        self.log(
            "Load checkpoint from file {}, "
            "Time usage:\n\tIO: {}, restore snapshot: {}".format(
                weights, t_io_end - t_start, t_end - t_io_end))


    def save_and_link_checkpoint(self, snapshot_dir):
        if self.local_rank > 0:
            return
        ensure_dir(snapshot_dir)
        current_iter_checkpoint = osp.join(
            snapshot_dir, 'iter-{}.pth'.format(self.state.iteration))
        self.save_checkpoint(current_iter_checkpoint)
        # last_iter_checkpoint = osp.join(
        #     snapshot_dir, 'iter-last.pth')
        # link_file(current_iter_checkpoint, last_iter_checkpoint)

    # def restore_checkpoint(self):
    #     self.load_checkpoint(self.continue_state_object, is_restore=True)

    def save_hdf5(self, path):
        if self.local_rank > 0:
            return
        save_dict = {}
        num_params = 0
        for k, v in self.state.model.state_dict().items():
            key = k
            if k.split('.')[0] == 'module':
                key = k[7:]
            np_array = v.cpu().numpy()
            save_dict[key] = np_array
            num_params += np_array.size
        if self.base_config is not None and self.base_config.deps is not None:
            save_dict['deps'] = self.base_config.deps
        save_hdf5(save_dict, path)
        print('---------------saved {} numpy arrays to {}---------------'.format(len(save_dict), path))
        self.log('num of params in hdf5={}'.format(num_params))

    def save_by_order(self, path):
        if self.local_rank > 0:
            return

        num_params = 0
        kernel_dict = {}
        sigma_dict = {}
        mu_dict = {}
        gamma_dict = {}
        beta_dict = {}
        other_dict = {}

        for k, v in self.state.model.state_dict().items():
            key = k
            if k.split('.')[0] == 'module':
                key = k[7:]
            np_array = v.cpu().numpy()
            num_params += np_array.size

            if 'conv.weight' in key:
                kernel_dict['kernel{}'.format(len(kernel_dict))] = np_array
            elif 'bn.weight' in key:
                gamma_dict['gamma{}'.format(len(gamma_dict))] = np_array
            elif 'bn.bias' in key:
                beta_dict['beta{}'.format(len(beta_dict))] = np_array
            elif 'bn.running_mean' in key:
                mu_dict['mu{}'.format(len(mu_dict))] = np_array
            elif 'bn.running_var' in key:
                sigma_dict['sigma{}'.format(len(sigma_dict))] = np_array
            else:
                other_dict[k] = np_array

        save_dict = {}
        save_dict.update(kernel_dict)
        save_dict.update(beta_dict)
        save_dict.update(gamma_dict)
        save_dict.update(mu_dict)
        save_dict.update(sigma_dict)
        save_dict.update(other_dict)
        if self.base_config is not None and self.base_config.deps is not None:
            save_dict['deps'] = self.base_config.deps
        save_hdf5(save_dict, path)
        print('---------------saved {} numpy arrays to {}---------------'.format(len(save_dict), path))
        self.log('num of params in hdf5={}'.format(num_params))

    def set_value(self, param, value):
        # assert tuple(param.size()) == tuple(value.shape)
        # if value.size != param.nelement():
        #     print('not equal: ', value.size, param.nelement)
        #     assert 0
        param.data = torch.from_numpy(value).cuda().type(torch.cuda.FloatTensor)

    def load_from_weights_dict(self, hdf5_dict, load_weights_keyword=None, path=None, ignore_keyword='IGNORE_KEYWORD'):
        assigned_params = 0
        for k, v in self.state.model.named_parameters():
            new_k = k.replace(ignore_keyword, '')
            if new_k in hdf5_dict and (load_weights_keyword is None or load_weights_keyword in new_k):
                self.echo('assign {} from hdf5'.format(k))
                # print(k, v.size(), hdf5_dict[k])
                self.set_value(v, hdf5_dict[new_k])
                assigned_params += 1
            else:
                self.echo('param {} not found in hdf5'.format(k))
        for k, v in self.state.model.named_buffers():
            new_k = k.replace(ignore_keyword, '')
            if new_k in hdf5_dict and (load_weights_keyword is None or load_weights_keyword in new_k):
                self.set_value(v, hdf5_dict[new_k])
                assigned_params += 1
            else:
                self.echo('buffer {} not found in hdf5'.format(k))
        msg = 'Assigned {} params '.format(assigned_params)
        if path is not None:
            msg += '  from hdf5: {}'.format(path)
        self.echo(msg)

    def load_hdf5(self, path, load_weights_keyword=None):
        hdf5_dict = read_hdf5(path)
        self.load_from_weights_dict(hdf5_dict, load_weights_keyword, path=path)

    def load_part(self, part_key, path):
        hdf5_dict = read_hdf5(path)
        self.load_from_weights_dict(hdf5_dict, load_weights_keyword=None, path=path, ignore_keyword=part_key)

    def load_by_order(self, path):
        hdf5_dict = read_hdf5(path)
        assigned_params = 0
        kernel_idx = 0
        sigma_idx = 0
        mu_idx = 0
        gamma_idx = 0
        beta_idx = 0
        for k, v in self.state.model.named_parameters():
            if k in hdf5_dict:
                value = hdf5_dict[k]
            else:
                if 'conv.weight' in k:
                    order_key = 'kernel{}'.format(kernel_idx)
                    kernel_idx += 1
                elif 'bn.weight' in k:
                    order_key = 'gamma{}'.format(gamma_idx)
                    gamma_idx += 1
                elif 'bn.bias' in k:
                    order_key = 'beta{}'.format(beta_idx)
                    beta_idx += 1
                else:
                    order_key = None
                value = None if order_key is None else hdf5_dict[order_key]
            if value is not None:
                self.set_value(v, value)
                assigned_params += 1

        for k, v in self.state.model.named_buffers():
            if k in hdf5_dict:
                value = hdf5_dict[k]
            else:
                if 'bn.running_mean' in k:
                    order_key = 'mu{}'.format(mu_idx)
                    mu_idx += 1
                elif 'bn.running_var' in k:
                    order_key = 'sigma{}'.format(sigma_idx)
                    sigma_idx += 1
                else:
                    order_key = None
                value = None if order_key is None else hdf5_dict[order_key]
            if value is not None:
                self.set_value(v, value)
                assigned_params += 1

        msg = 'Assigned {} params '.format(assigned_params)
        if path is not None:
            msg += '  from hdf5: {}'.format(path)
        self.echo(msg)





    def log(self, msg):
        if self.local_rank == 0:
            print(msg)
            self.logger.info(msg)

    def save_latest_ckpt(self, snapshot_dir):
        if self.local_rank > 0:
            return
        ensure_dir(snapshot_dir)
        current_iter_checkpoint = osp.join(
            snapshot_dir, 'latest.pth')
        self.save_checkpoint(current_iter_checkpoint)

    def show_variables(self):
        if self.local_rank == 0:
            print('---------- show variables -------------')
            num_params = 0
            for k, v in self.state.model.state_dict().items():
                print(k, v.shape)
                num_params += v.nelement()
            print('num params: ', num_params)
            print('--------------------------------------')


    def echo(self, msg):
        if self.local_rank == 0:
            print(msg)

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            if self.logger is not None:
                self.logger.warning(
                    "A exception occurred during Engine initialization, "
                    "give up running process")
            return False

    def __enter__(self):
        return self