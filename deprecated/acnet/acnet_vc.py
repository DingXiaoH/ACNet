from ding_train import ding_train
from base_config import get_baseconfig_by_epoch
from utils.misc import start_exp
from constants import VGG_ORIGIN_DEPS, parse_usual_lr_schedule

def acnet_vc():
    try_arg = start_exp()

    network_type = 'vc'
    dataset_name = 'cifar10'
    log_dir = 'acnet_exps/{}_{}_train'.format(network_type, try_arg)
    save_weights = 'acnet_exps/{}_{}_savedweights.pth'.format(network_type, try_arg)
    weight_decay_strength = 1e-4
    batch_size = 64
    deps = VGG_ORIGIN_DEPS

    lrs = parse_usual_lr_schedule(try_arg)

    if 'bias' in try_arg:
        weight_decay_bias = weight_decay_strength
    else:
        weight_decay_bias = 0



    if 'warmup' in try_arg:
        warmup_factor = 0
    else:
        warmup_factor = 1

    config = get_baseconfig_by_epoch(network_type=network_type, dataset_name=dataset_name, dataset_subset='train',
                                     global_batch_size=batch_size, num_node=1,
                                     weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=0.9,
                                     max_epochs=lrs.max_epochs, base_lr=lrs.base_lr, lr_epoch_boundaries=lrs.lr_epoch_boundaries,
                                     lr_decay_factor=lrs.lr_decay_factor,
                                     warmup_epochs=5, warmup_method='linear', warmup_factor=warmup_factor,
                                     ckpt_iter_period=20000, tb_iter_period=100, output_dir=log_dir,
                                     tb_dir=log_dir, save_weights=save_weights, val_epoch_period=2, linear_final_lr=lrs.linear_final_lr,
                                     weight_decay_bias=weight_decay_bias, deps=deps)

    if 'normal' in try_arg:
        builder = None
    elif 'acnet' in try_arg:
        from acnet.acnet_builder import ACNetBuilder
        builder = ACNetBuilder(base_config=config, deploy=False)
    else:
        assert False

    ding_train(config, show_variables=True, convbuilder=builder, use_nesterov='nest' in try_arg)



if __name__ == '__main__':
    acnet_vc()