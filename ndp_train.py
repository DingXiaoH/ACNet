from torch.utils.tensorboard import SummaryWriter
from base_config import BaseConfigByEpoch
from model_map import get_model_fn
from data.data_factory import create_dataset, load_cuda_data, num_iters_per_epoch
from torch.nn.modules.loss import CrossEntropyLoss
from utils.engine import Engine
from utils.pyt_utils import ensure_dir
from utils.misc import torch_accuracy, AvgMeter
from collections import OrderedDict
import torch
from tqdm import tqdm
import time
from builder import ConvBuilder
from utils.lr_scheduler import get_lr_scheduler
import os
from utils.checkpoint import get_last_checkpoint
from utils.misc import init_as_tensorflow
from ndp_test import val_during_train

TRAIN_SPEED_START = 0.1
TRAIN_SPEED_END = 0.2

COLLECT_TRAIN_LOSS_EPOCHS = 3

TEST_BATCH_SIZE = 100

def train_one_step(net, data, label, optimizer, criterion,
                   if_accum_grad = False, gradient_mask_tensor = None, lasso_keyword_to_strength=None):
    pred = net(data)
    loss = criterion(pred, label)

    if lasso_keyword_to_strength is not None:
        assert len(lasso_keyword_to_strength) == 1 #TODO
        for lasso_key, lasso_strength in lasso_keyword_to_strength.items():
            for name, param in net.named_parameters():
                if lasso_key in name:
                    if param.ndimension() == 1:
                        loss += lasso_strength * param.abs().sum()
                        # print('lasso on vec ', name)
                    else:
                        assert param.ndimension() == 4
                        loss += lasso_strength * ((param ** 2).sum(dim=(1, 2, 3)).sqrt().sum())
                        # print('lasso on tensor ', name)

    loss.backward()
    if not if_accum_grad:
        if gradient_mask_tensor is not None:
            for name, param in net.named_parameters():
                if name in gradient_mask_tensor:
                    param.grad = param.grad * gradient_mask_tensor[name]
        optimizer.step()
        optimizer.zero_grad()
    acc, acc5 = torch_accuracy(pred, label, (1,5))
    return acc, acc5, loss


def sgd_optimizer(engine, cfg, model, no_l2_keywords, use_nesterov, keyword_to_lr_mult):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.base_lr
        weight_decay = cfg.weight_decay
        if "bias" in key or "bn" in key or "BN" in key:
            # lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.weight_decay_bias
            engine.echo('set weight_decay_bias={} for {}'.format(weight_decay, key))
        for kw in no_l2_keywords:
            if kw in key:
                weight_decay = 0
                engine.echo('NOTICE! weight decay = 0 for {} because {} in {}'.format(key, kw, key))
                break
        if 'bias' in key:
            apply_lr = 2 * lr
        else:
            apply_lr = lr
        if keyword_to_lr_mult is not None:
            for keyword, mult in keyword_to_lr_mult.items():
                if keyword in key:
                    apply_lr *= mult
                    engine.echo('multiply lr of {} by {}'.format(key, mult))
                    break
        params += [{"params": [value], "lr": apply_lr, "weight_decay": weight_decay}]
    # optimizer = torch.optim.Adam(params, lr)
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.momentum, nesterov=use_nesterov)
    return optimizer

def get_optimizer(engine, cfg, model, no_l2_keywords, use_nesterov=False, keyword_to_lr_mult=None):
    return sgd_optimizer(engine, cfg, model, no_l2_keywords, use_nesterov=use_nesterov, keyword_to_lr_mult=keyword_to_lr_mult)

def get_criterion(cfg):
    return CrossEntropyLoss()


def train_main(
        local_rank,
        cfg:BaseConfigByEpoch, net=None, train_dataloader=None, val_dataloader=None, show_variables=False, convbuilder=None,
               init_hdf5=None, no_l2_keywords='depth', gradient_mask=None, use_nesterov=False, tensorflow_style_init=False,
               load_weights_keyword=None,
               keyword_to_lr_mult=None,
               auto_continue=False,
lasso_keyword_to_strength=None,
save_hdf5_epochs=10000):

    if no_l2_keywords is None:
        no_l2_keywords = []
    if type(no_l2_keywords) is not list:
        no_l2_keywords = [no_l2_keywords]

    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.tb_dir)
    with Engine(local_rank=local_rank) as engine:
        engine.setup_log(
            name='train', log_dir=cfg.output_dir, file_name='log.txt')

        # ----------------------------- build model ------------------------------
        if convbuilder is None:
            convbuilder = ConvBuilder(base_config=cfg)
        if net is None:
            net_fn = get_model_fn(cfg.dataset_name, cfg.network_type)
            model = net_fn(cfg, convbuilder)
        else:
            model = net
        model = model.cuda()
        # ----------------------------- model done ------------------------------

        # ---------------------------- prepare data -------------------------
        if train_dataloader is None:
            train_data = create_dataset(cfg.dataset_name, cfg.dataset_subset,
                                        cfg.global_batch_size, distributed=engine.distributed)
        if cfg.val_epoch_period > 0 and val_dataloader is None:
            val_data = create_dataset(cfg.dataset_name, 'val',
                                      global_batch_size=100, distributed=False)
        engine.echo('NOTE: Data prepared')
        engine.echo('NOTE: We have global_batch_size={} on {} GPUs, the allocated GPU memory is {}'.format(cfg.global_batch_size, torch.cuda.device_count(), torch.cuda.memory_allocated()))
        # ----------------------------- data done --------------------------------

        # ------------------------ parepare optimizer, scheduler, criterion -------
        optimizer = get_optimizer(engine, cfg, model,
                                  no_l2_keywords=no_l2_keywords, use_nesterov=use_nesterov, keyword_to_lr_mult=keyword_to_lr_mult)
        scheduler = get_lr_scheduler(cfg, optimizer)
        criterion = get_criterion(cfg).cuda()
        # --------------------------------- done -------------------------------

        engine.register_state(
            scheduler=scheduler, model=model, optimizer=optimizer)

        if engine.distributed:
            torch.cuda.set_device(local_rank)
            engine.echo('Distributed training, device {}'.format(local_rank))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank],
                broadcast_buffers=False, )
        else:
            assert torch.cuda.device_count() == 1
            engine.echo('Single GPU training')

        if tensorflow_style_init:
            init_as_tensorflow(model)
        if cfg.init_weights:
            engine.load_checkpoint(cfg.init_weights)
        if init_hdf5:
            engine.load_hdf5(init_hdf5, load_weights_keyword=load_weights_keyword)
        if auto_continue:
            assert cfg.init_weights is None
            engine.load_checkpoint(get_last_checkpoint(cfg.output_dir))
        if show_variables:
            engine.show_variables()

        # ------------ do training ---------------------------- #
        engine.log("\n\nStart training with pytorch version {}".format(torch.__version__))

        iteration = engine.state.iteration
        iters_per_epoch = num_iters_per_epoch(cfg)
        max_iters = iters_per_epoch * cfg.max_epochs
        tb_writer = SummaryWriter(cfg.tb_dir)
        tb_tags = ['Top1-Acc', 'Top5-Acc', 'Loss']

        model.train()

        done_epochs = iteration // iters_per_epoch
        last_epoch_done_iters = iteration % iters_per_epoch

        if done_epochs == 0 and last_epoch_done_iters == 0:
            engine.save_hdf5(os.path.join(cfg.output_dir, 'init.hdf5'))

        recorded_train_time = 0
        recorded_train_examples = 0

        collected_train_loss_sum = 0
        collected_train_loss_count = 0

        if gradient_mask is not None:
            gradient_mask_tensor = {}
            for name, value in gradient_mask.items():
                gradient_mask_tensor[name] = torch.Tensor(value).cuda()
        else:
            gradient_mask_tensor = None

        for epoch in range(done_epochs, cfg.max_epochs):

            if engine.distributed and hasattr(train_data, 'train_sampler'):
                train_data.train_sampler.set_epoch(epoch)

            if epoch == done_epochs:
                pbar = tqdm(range(iters_per_epoch - last_epoch_done_iters))
            else:
                pbar = tqdm(range(iters_per_epoch))

            if epoch == 0 and local_rank == 0:
                val_during_train(epoch=epoch, iteration=iteration, tb_tags=tb_tags, engine=engine, model=model,
                                 val_data=val_data, criterion=criterion, descrip_str='Init',
                                 dataset_name=cfg.dataset_name, test_batch_size=TEST_BATCH_SIZE,
                                 tb_writer=tb_writer)


            top1 = AvgMeter()
            top5 = AvgMeter()
            losses = AvgMeter()
            discrip_str = 'Epoch-{}/{}'.format(epoch, cfg.max_epochs)
            pbar.set_description('Train' + discrip_str)

            for _ in pbar:

                start_time = time.time()
                data, label = load_cuda_data(train_data, dataset_name=cfg.dataset_name)

                    # load_cuda_data(train_dataloader, cfg.dataset_name)
                data_time = time.time() - start_time

                if_accum_grad = ((iteration % cfg.grad_accum_iters) != 0)

                train_net_time_start = time.time()
                acc, acc5, loss = train_one_step(model, data, label, optimizer, criterion,
                                                 if_accum_grad, gradient_mask_tensor=gradient_mask_tensor,
                                                 lasso_keyword_to_strength=lasso_keyword_to_strength)
                train_net_time_end = time.time()

                if iteration > TRAIN_SPEED_START * max_iters and iteration < TRAIN_SPEED_END * max_iters:
                    recorded_train_examples += cfg.global_batch_size
                    recorded_train_time += train_net_time_end - train_net_time_start

                scheduler.step()

                for module in model.modules():
                    if hasattr(module, 'set_cur_iter'):
                        module.set_cur_iter(iteration)

                if iteration % cfg.tb_iter_period == 0 and engine.world_rank == 0:
                    for tag, value in zip(tb_tags, [acc.item(), acc5.item(), loss.item()]):
                        tb_writer.add_scalars(tag, {'Train': value}, iteration)

                top1.update(acc.item())
                top5.update(acc5.item())
                losses.update(loss.item())

                if epoch >= cfg.max_epochs - COLLECT_TRAIN_LOSS_EPOCHS:
                    collected_train_loss_sum += loss.item()
                    collected_train_loss_count += 1

                pbar_dic = OrderedDict()
                pbar_dic['data-time'] = '{:.2f}'.format(data_time)
                pbar_dic['cur_iter'] = iteration
                pbar_dic['lr'] = scheduler.get_lr()[0]
                pbar_dic['top1'] = '{:.5f}'.format(top1.mean)
                pbar_dic['top5'] = '{:.5f}'.format(top5.mean)
                pbar_dic['loss'] = '{:.5f}'.format(losses.mean)
                pbar.set_postfix(pbar_dic)

                iteration += 1

                if iteration >= max_iters or iteration % cfg.ckpt_iter_period == 0:
                    engine.update_iteration(iteration)
                    if (not engine.distributed) or (engine.distributed and engine.world_rank == 0):
                        engine.save_and_link_checkpoint(cfg.output_dir)

                if iteration >= max_iters:
                    break

            #   do something after an epoch?
            engine.update_iteration(iteration)
            engine.save_latest_ckpt(cfg.output_dir)

            if (epoch + 1) % save_hdf5_epochs == 0:
                engine.save_hdf5(os.path.join(cfg.output_dir, 'epoch-{}.hdf5'.format(epoch)))

            if local_rank == 0 and \
                    cfg.val_epoch_period > 0 and (epoch >= cfg.max_epochs - 10 or epoch % cfg.val_epoch_period == 0):
                val_during_train(epoch=epoch, iteration=iteration, tb_tags=tb_tags, engine=engine, model=model,
                                 val_data=val_data, criterion=criterion, descrip_str=discrip_str,
                                 dataset_name=cfg.dataset_name, test_batch_size=TEST_BATCH_SIZE, tb_writer=tb_writer)

            if iteration >= max_iters:
                break

        #   do something after the training
        if recorded_train_time > 0:
            exp_per_sec = recorded_train_examples / recorded_train_time
        else:
            exp_per_sec = 0
        engine.log(
            'TRAIN speed: from {} to {} iterations, batch_size={}, examples={}, total_net_time={:.4f}, examples/sec={}'
            .format(int(TRAIN_SPEED_START * max_iters), int(TRAIN_SPEED_END * max_iters), cfg.global_batch_size,
                    recorded_train_examples, recorded_train_time, exp_per_sec))
        if cfg.save_weights:
            engine.save_checkpoint(cfg.save_weights)
            print('NOTE: training finished, saved to {}'.format(cfg.save_weights))
        engine.save_hdf5(os.path.join(cfg.output_dir, 'finish.hdf5'))
        if collected_train_loss_count > 0:
            engine.log('TRAIN LOSS collected over last {} epochs: {:.6f}'.format(COLLECT_TRAIN_LOSS_EPOCHS,
                                                                             collected_train_loss_sum / collected_train_loss_count))