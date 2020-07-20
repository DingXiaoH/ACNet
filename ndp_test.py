from base_config import BaseConfigByEpoch
from model_map import get_model_fn
from data.data_factory import create_dataset, load_cuda_data
from torch.nn.modules.loss import CrossEntropyLoss
from utils.engine import Engine
from utils.misc import torch_accuracy, AvgMeter
from collections import OrderedDict
import torch
from tqdm import tqdm
import time
from builder import ConvBuilder
from utils.misc import log_important, extract_deps_from_weights_file
from base_config import get_baseconfig_for_test
from data.data_factory import num_val_examples

SPEED_TEST_SAMPLE_IGNORE_RATIO = 0.5

TEST_BATCH_SIZE = 100
OVERALL_LOG_FILE = 'overall_test_log.txt'
DETAIL_LOG_FILE = 'detail_test_log.txt'

def run_eval(val_data, max_iters, net, criterion, discrip_str, dataset_name):
    pbar = tqdm(range(max_iters))
    top1 = AvgMeter()
    top5 = AvgMeter()
    losses = AvgMeter()
    pbar.set_description('Validation' + discrip_str)
    total_net_time = 0
    with torch.no_grad():
        for iter_idx, i in enumerate(pbar):
            start_time = time.time()

            data, label = load_cuda_data(val_data, dataset_name=dataset_name)
            data_time = time.time() - start_time

            net_time_start = time.time()
            pred = net(data)
            net_time_end = time.time()

            if iter_idx >= SPEED_TEST_SAMPLE_IGNORE_RATIO * max_iters:
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
    # reduced_metirc_dic = reduce_loss_dict(metric_dic)
    reduced_metirc_dic = metric_dic     #TODO note this
    return reduced_metirc_dic, total_net_time


def val_during_train(epoch, iteration, tb_tags,
                      engine, model, val_data, criterion, descrip_str,
                      dataset_name, test_batch_size, tb_writer):
    model.eval()
    num_examples = num_val_examples(dataset_name)
    assert num_examples % test_batch_size == 0
    val_iters = num_examples // test_batch_size
    eval_dict, total_net_time = run_eval(val_data, val_iters, model, criterion, descrip_str,
                                         dataset_name=dataset_name)
    val_top1_value = eval_dict['top1'].item()
    val_top5_value = eval_dict['top5'].item()
    val_loss_value = eval_dict['loss'].item()
    for tag, value in zip(tb_tags, [val_top1_value, val_top5_value, val_loss_value]):
        tb_writer.add_scalars(tag, {'Val': value}, iteration)
    engine.log(
        'val at epoch {}, top1={:.5f}, top5={:.5f}, loss={:.6f}'.format(epoch, val_top1_value,
                                                                             val_top5_value,
                                                                             val_loss_value))
    model.train()


def get_criterion(cfg):
    return CrossEntropyLoss()   #TODO note this


def ding_test(cfg:BaseConfigByEpoch, net=None, val_dataloader=None, show_variables=False, convbuilder=None,
               init_hdf5=None, extra_msg=None, weights_dict=None):

    with Engine(local_rank=0, for_val_only=True) as engine:

        engine.setup_log(
            name='test', log_dir='./', file_name=DETAIL_LOG_FILE)

        if convbuilder is None:
            convbuilder = ConvBuilder(base_config=cfg)

        if net is None:
            net_fn = get_model_fn(cfg.dataset_name, cfg.network_type)
            model = net_fn(cfg, convbuilder).cuda()
        else:
            model = net.cuda()

        if val_dataloader is None:
            val_data = create_dataset(cfg.dataset_name, cfg.dataset_subset,
                                      global_batch_size=cfg.global_batch_size, distributed=False)
        num_examples = num_val_examples(cfg.dataset_name)
        assert num_examples % cfg.global_batch_size == 0
        val_iters = num_val_examples(cfg.dataset_name) // cfg.global_batch_size
        print('batchsize={}, {} iters'.format(cfg.global_batch_size, val_iters))

        criterion = get_criterion(cfg).cuda()

        engine.register_state(
            scheduler=None, model=model, optimizer=None)

        if show_variables:
            engine.show_variables()

        assert not engine.distributed

        if weights_dict is not None:
            engine.load_from_weights_dict(weights_dict)
        else:
            if cfg.init_weights:
                engine.load_checkpoint(cfg.init_weights)
            if init_hdf5:
                engine.load_hdf5(init_hdf5)

        # engine.save_by_order('smi2_by_order.hdf5')
        # engine.load_by_order('smi2_by_order.hdf5')
        # engine.save_hdf5('model_files/stami2_lrs4Z.hdf5')

        model.eval()
        eval_dict, total_net_time = run_eval(val_data, val_iters, model, criterion, 'TEST', dataset_name=cfg.dataset_name)
        val_top1_value = eval_dict['top1'].item()
        val_top5_value = eval_dict['top5'].item()
        val_loss_value = eval_dict['loss'].item()

        msg = '{},{},{},top1={:.5f},top5={:.5f},loss={:.7f},total_net_time={}'.format(cfg.network_type, init_hdf5 or cfg.init_weights, cfg.dataset_subset,
                                                                    val_top1_value, val_top5_value, val_loss_value, total_net_time)
        if extra_msg is not None:
            msg += ', ' + extra_msg
        log_important(msg, OVERALL_LOG_FILE)
        return eval_dict


def general_test(network_type, weights, builder=None, net=None, dataset_name=None, weights_dict=None,
                 batch_size=None):
    if weights is None or weights == 'None':
        init_weights = None
        init_hdf5 = None
    elif weights.endswith('.hdf5'):
        init_weights = None
        init_hdf5 = weights
    else:
        init_weights = weights
        init_hdf5 = None

    if init_hdf5 is not None:
        deps = extract_deps_from_weights_file(init_hdf5)
    else:
        deps = None

    if batch_size is None:
        batch_size = TEST_BATCH_SIZE
    test_config = get_baseconfig_for_test(network_type=network_type, dataset_subset='val', global_batch_size=batch_size,
                                          init_weights=init_weights, deps=deps, dataset_name=dataset_name)
    return ding_test(cfg=test_config, net=net, show_variables=True, init_hdf5=init_hdf5, convbuilder=builder, weights_dict=weights_dict)


if __name__ == '__main__':
    import sys
    builder = None
    if 'deploy' in sys.argv[2]:
        from nobn_builder import NoBNBuilder
        builder = NoBNBuilder(base_config=None)
    general_test(network_type=sys.argv[1], weights=sys.argv[2], builder=builder)