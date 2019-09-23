from torch.utils.tensorboard import SummaryWriter
from base_config import BaseConfigByEpoch
from base_model.model_map import get_model_fn
from dataset import create_dataset, num_train_examples_per_epoch
from torch.nn.modules.loss import CrossEntropyLoss
from utils.engine import Engine
from utils.pyt_utils import ensure_dir
from utils.misc import torch_accuracy, AvgMeter
from utils.comm import reduce_loss_dict
from collections import OrderedDict
import torch
from tqdm import tqdm
import time
from builder import ConvBuilder
from utils.lr_scheduler import WarmupMultiStepLR

# try:
#     from apex.parallel.distributed import DistributedDataParallel
#     from apex import amp
# except ImportError:
#     raise ImportError('Use APEX for multi-precision via apex.amp')

def train_one_step(net, data, label, optimizer, criterion, if_accum_grad = False):
    pred = net(data)
    loss = criterion(pred, label)
    loss.backward()
    if not if_accum_grad:
        optimizer.step()
        optimizer.zero_grad()
    acc, acc5 = torch_accuracy(pred, label, (1,5))
    return acc, acc5, loss

def load_cuda_data(data_loader, dataset_name):
    if dataset_name == 'imagenet':
        data_dict = next(data_loader)
        data = data_dict['data']
        label = data_dict['label']
        data = torch.from_numpy(data).cuda()
        label = torch.from_numpy(label).type(torch.long).cuda()
    else:
        data, label = next(data_loader)
        data = data.cuda()
        label = label.cuda()
    return data, label

def run_eval(ds_val, max_iters, net, criterion, discrip_str, dataset_name):
    pbar = tqdm(range(max_iters))
    top1 = AvgMeter()
    top5 = AvgMeter()
    losses = AvgMeter()
    pbar.set_description('Validation' + discrip_str)
    with torch.no_grad():
        for i in pbar:
            start_time = time.time()
            data, label = load_cuda_data(ds_val, dataset_name=dataset_name)
            data_time = time.time() - start_time

            pred = net(data)
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
    # reduced_metirc_dic = my_reduce_dic(metric_dic)
    return reduced_metirc_dic


def sgd_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.base_lr
        weight_decay = cfg.weight_decay
        if "bias" in key or "bn" in key or "BN" in key:
            # lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.weight_decay_bias
            print('set weight_decay_bias={} for {}'.format(weight_decay, key))
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.momentum)
    return optimizer

def get_optimizer(cfg, model):
    return sgd_optimizer(cfg, model)

def get_criterion(cfg):
    return CrossEntropyLoss()

def num_iters_per_epoch(cfg):
    return num_train_examples_per_epoch(cfg.dataset_name) // cfg.global_batch_size

#   LR scheduler should work according the number of iterations
def get_lr_scheduler(cfg, optimizer):
    it_ep = num_iters_per_epoch(cfg)
    lr_iter_boundaries = [it_ep * ep for ep in cfg.lr_epoch_boundaries]
    return WarmupMultiStepLR(
            optimizer, lr_iter_boundaries, cfg.lr_decay_factor,
            warmup_factor=cfg.warmup_factor,
            warmup_iters=cfg.warmup_epochs * it_ep,
            warmup_method=cfg.warmup_method, )


def ding_train(cfg:BaseConfigByEpoch, net=None, train_dataloader=None, val_dataloader=None, show_variables=False, convbuilder=None):

    # LOCAL_RANK = 0
    #
    # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # is_distributed = num_gpus > 1
    #
    # if is_distributed:
    #     torch.cuda.set_device(LOCAL_RANK)
    #     torch.distributed.init_process_group(
    #         backend="nccl", init_method="env://"
    #     )
    #     synchronize()
    #
    # torch.backends.cudnn.benchmark = True

    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.tb_dir)
    with Engine(cfg) as engine:

        is_main_process = (engine.world_rank == 0) #TODO correct?

        logger = engine.setup_log(
            name='train', log_dir=cfg.output_dir, file_name='log.txt')

        # -- typical model components model, opt,  scheduler,  dataloder --#
        if net is None:
            net = get_model_fn(cfg.dataset_name, cfg.network_type)

        if convbuilder is None:
            convbuilder = ConvBuilder()

        model = net(cfg, convbuilder).cuda()

        if train_dataloader is None:
            train_dataloader = create_dataset(cfg.dataset_name, cfg.dataset_subset, cfg.global_batch_size)
        if cfg.val_epoch_period > 0 and val_dataloader is None:
            val_dataloader = create_dataset(cfg.dataset_name, 'val', batch_size=100)    #TODO 100?

        print('NOTE: Data prepared')
        print('NOTE: We have global_batch_size={} on {} GPUs, the allocated GPU memory is {}'.format(cfg.global_batch_size, torch.cuda.device_count(), torch.cuda.memory_allocated()))

        # device = torch.device(cfg.device)
        # model.to(device)
        # model.cuda()

        optimizer = get_optimizer(cfg, model)
        scheduler = get_lr_scheduler(cfg, optimizer)
        criterion = get_criterion(cfg).cuda()

        # model, optimizer = amp.initialize(model, optimizer, opt_level="O0")

        engine.register_state(
            scheduler=scheduler, model=model, optimizer=optimizer)

        if engine.distributed:
            print('Distributed training, engine.world_rank={}'.format(engine.world_rank))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[engine.world_rank],
                broadcast_buffers=False, )
            # model = DistributedDataParallel(model, delay_allreduce=True)

        if engine.continue_state_object:
            engine.restore_checkpoint()
        else:
            if cfg.init_weights:
                engine.load_checkpoint(cfg.init_weights, is_restore=False)

        if show_variables:
            engine.show_variables()

        # ------------ do training ---------------------------- #
        logger.info("\n\nStart training with pytorch version {}".format(torch.__version__))

        iteration = engine.state.iteration
        # done_epochs = iteration // num_train_examples_per_epoch(cfg.dataset_name)
        iters_per_epoch = num_iters_per_epoch(cfg)
        max_iters = iters_per_epoch * cfg.max_epochs
        tb_writer = SummaryWriter(cfg.tb_dir)
        tb_tags = ['Top1-Acc', 'Top5-Acc', 'Loss']

        model.train()

        done_epochs = iteration // iters_per_epoch

        for epoch in range(done_epochs, cfg.max_epochs):

            pbar = tqdm(range(iters_per_epoch))
            top1 = AvgMeter()
            top5 = AvgMeter()
            losses = AvgMeter()
            discrip_str = 'Epoch-{}/{}'.format(epoch, cfg.max_epochs)
            pbar.set_description('Train' + discrip_str)

            if cfg.val_epoch_period > 0 and epoch % cfg.val_epoch_period == 0:
                model.eval()
                val_iters = 500 if cfg.dataset_name == 'imagenet' else 100  # use batch_size=100 for val on ImagenNet and CIFAR
                eval_dict = run_eval(val_dataloader, val_iters, model, criterion, discrip_str, dataset_name=cfg.dataset_name)
                val_top1_value = eval_dict['top1'].item()
                val_top5_value = eval_dict['top5'].item()
                val_loss_value = eval_dict['loss'].item()
                for tag, value in zip(tb_tags, [val_top1_value, val_top5_value, val_loss_value]):
                    tb_writer.add_scalars(tag, {'Val': value}, iteration)
                engine.log('validate at epoch {}, top1={:.5f}, top5={:.5f}, loss={:.6f}'.format(epoch, val_top1_value, val_top5_value, val_loss_value))
                model.train()

            for _ in pbar:

                scheduler.step()

                start_time = time.time()
                data, label = load_cuda_data(train_dataloader, cfg.dataset_name)
                data_time = time.time() - start_time

                if_accum_grad = ((iteration % cfg.grad_accum_iters) != 0)
                acc, acc5, loss = train_one_step(model, data, label, optimizer, criterion, if_accum_grad)

                if iteration % cfg.tb_iter_period == 0 and is_main_process:
                    for tag, value in zip(tb_tags, [acc.item(), acc5.item(), loss.item()]):
                        tb_writer.add_scalars(tag, {'Train': value}, iteration)

                top1.update(acc.item())
                top5.update(acc5.item())
                losses.update(loss.item())

                pbar_dic = OrderedDict()
                pbar_dic['data-time'] = '{:.2f}'.format(data_time)
                pbar_dic['cur_iter'] = iteration
                pbar_dic['lr'] = scheduler.get_lr()[0]
                pbar_dic['top1'] = '{:.5f}'.format(top1.mean)
                pbar_dic['top5'] = '{:.5f}'.format(top5.mean)
                pbar_dic['loss'] = '{:.5f}'.format(losses.mean)
                pbar.set_postfix(pbar_dic)

                if iteration >= max_iters or iteration % cfg.ckpt_iter_period == 0:
                    engine.update_iteration(iteration)
                    if (not engine.distributed) or (engine.distributed and is_main_process):
                        engine.save_and_link_checkpoint(cfg.output_dir)

                iteration += 1
                if iteration >= max_iters:
                    break

            #   do something after an epoch?
            if iteration >= max_iters:
                break
        #   do something after the training
        engine.save_checkpoint(cfg.save_weights)
        print('NOTE: training finished, saved to {}'.format(cfg.save_weights))





            # if engine.world_rank == 0:
            #     if iteration % 20 == 0 or iteration == max_iter:
            #         # loss_dict = reducke_loss_dict(loss_dict)
            #         log_str = 'it:%d, lr:%.1e, ' % (
            #             iteration, optimizer.param_groups[0]["lr"])
            #         for key in loss_dict:
            #             tb_writer.add_scalar(
            #                 key, loss_dict[key].mean(), global_step=iteration)
            #             log_str += key + ': %.3f, ' % float(loss_dict[key])
            #         logger.info(log_str)
