from base_config import BaseConfigByEpoch
from model_map import get_model_fn
from dataset import create_dataset
from torch.nn.modules.loss import CrossEntropyLoss
from utils.engine import Engine
from utils.misc import torch_accuracy, AvgMeter
from utils.comm import reduce_loss_dict
from collections import OrderedDict
import torch
from tqdm import tqdm
import time
from builder import ConvBuilder
from ding_train import load_cuda_data
import sys
from utils.misc import log_important
from base_config import get_baseconfig_for_test

TEST_BATCH_SIZE = 100
OVERALL_LOG_FILE = 'overall_test_log.txt'
DETAIL_LOG_FILE = 'detail_test_log.txt'

def run_eval(ds_val, max_iters, net, criterion, discrip_str, dataset_name):
    pbar = tqdm(range(max_iters))
    top1 = AvgMeter()
    top5 = AvgMeter()
    losses = AvgMeter()
    pbar.set_description('Validation' + discrip_str)
    total_net_time = 0
    with torch.no_grad():
        for iter_idx, i in enumerate(pbar):
            start_time = time.time()
            data, label = load_cuda_data(ds_val, dataset_name=dataset_name)
            data_time = time.time() - start_time

            net_time_start = time.time()
            pred = net(data)
            net_time_end = time.time()
            total_net_time += net_time_end - net_time_start

            loss = criterion(pred, label)
            acc, acc5 = torch_accuracy(pred, label, (1, 5))

            top1.update(acc.item())
            top5.update(acc5.item())
            losses.update(loss.item())
            pbar_dic = OrderedDict()
            pbar_dic['data-time'] = '{:.2f}'.format(data_time)
            pbar_dic['top1'] = '{:.5f}'.format(top1.mean)
            pbar_dic['top5'] = '{:.5f}'.format(top5.mean)
            pbar_dic['loss'] = '{:.5f}'.format(losses.mean)
            pbar.set_postfix(pbar_dic)

    metric_dic = {'top1':torch.tensor(top1.mean),
                  'top5':torch.tensor(top5.mean),
                  'loss':torch.tensor(losses.mean)}
    reduced_metirc_dic = reduce_loss_dict(metric_dic)
    return reduced_metirc_dic, total_net_time

def get_criterion(cfg):
    return CrossEntropyLoss()

def ding_test(cfg:BaseConfigByEpoch, net=None, val_dataloader=None, show_variables=False, convbuilder=None,
               init_hdf5=None, ):

    with Engine() as engine:

        engine.setup_log(
            name='test', log_dir='./', file_name=DETAIL_LOG_FILE)

        if net is None:
            net = get_model_fn(cfg.dataset_name, cfg.network_type)

        if convbuilder is None:
            convbuilder = ConvBuilder(base_config=cfg)

        model = net(cfg, convbuilder).cuda()

        if val_dataloader is None:
            val_dataloader = create_dataset(cfg.dataset_name, cfg.dataset_subset, batch_size=cfg.global_batch_size)
        val_iters = 50000 // cfg.global_batch_size if cfg.dataset_name == 'imagenet' else 10000 // cfg.global_batch_size

        print('NOTE: Data prepared')
        print('NOTE: We have global_batch_size={} on {} GPUs, the allocated GPU memory is {}'.format(cfg.global_batch_size, torch.cuda.device_count(), torch.cuda.memory_allocated()))

        criterion = get_criterion(cfg).cuda()

        engine.register_state(
            scheduler=None, model=model, optimizer=None)

        if engine.distributed:
            print('Distributed training, engine.world_rank={}'.format(engine.world_rank))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[engine.world_rank],
                broadcast_buffers=False, )
            # model = DistributedDataParallel(model, delay_allreduce=True)
        elif torch.cuda.device_count() > 1:
            print('Single machine multiple GPU training')
            model = torch.nn.parallel.DataParallel(model)

        if cfg.init_weights:
            engine.load_checkpoint(cfg.init_weights, is_restore=True, just_weights=True)

        if init_hdf5:
            engine.load_hdf5(init_hdf5)

        if show_variables:
            engine.show_variables()

        model.eval()
        eval_dict, _ = run_eval(val_dataloader, val_iters, model, criterion, 'TEST', dataset_name=cfg.dataset_name)
        val_top1_value = eval_dict['top1'].item()
        val_top5_value = eval_dict['top5'].item()
        val_loss_value = eval_dict['loss'].item()

        msg = '{},{},{},top1={:.5f},top5={:.5f},loss={:.7f}'.format(cfg.network_type, init_hdf5 or cfg.init_weights, cfg.dataset_subset,
                                                                    val_top1_value, val_top5_value, val_loss_value)
        log_important(msg, OVERALL_LOG_FILE)


def general_test(network_type, weights, builder=None):
    if weights.endswith('.hdf5'):
        init_weights = None
        init_hdf5 = weights
    else:
        init_weights = weights
        init_hdf5 = None
    if 'wrnc16' in network_type or 'wrnh16' in network_type:
        from constants import wrn_origin_deps_flattened
        deps = wrn_origin_deps_flattened(2, 8)
    else:
        deps = None
    test_config = get_baseconfig_for_test(network_type=network_type, dataset_subset='val', global_batch_size=TEST_BATCH_SIZE,
                                          init_weights=init_weights, deps=deps)
    ding_test(cfg=test_config, show_variables=True, init_hdf5=init_hdf5, convbuilder=builder)



if __name__ == '__main__':
    network_type = sys.argv[1]
    weights = sys.argv[2]
    general_test(network_type=network_type, weights=weights)


