from collections import namedtuple
from model_map import get_dataset_name_by_model_name

BaseConfigByEpoch = namedtuple('BaseConfigByEpoch', ['network_type', 'dataset_name', 'dataset_subset', 'global_batch_size', 'num_node', 'device',
                                       'weight_decay', 'weight_decay_bias', 'optimizer_type', 'momentum',
                                       'bias_lr_factor', 'max_epochs', 'base_lr', 'lr_epoch_boundaries', 'lr_decay_factor', 'linear_final_lr',
                                       'warmup_epochs', 'warmup_method', 'warmup_factor',
                                       'ckpt_iter_period', 'tb_iter_period',
                                       'output_dir',  'tb_dir',
                                       'init_weights', 'save_weights',
                                       'val_epoch_period', 'grad_accum_iters',
                                                     'deps',
                                                     'se_reduce_scale'])

def get_baseconfig_by_epoch(network_type, dataset_name, dataset_subset, global_batch_size, num_node,
                    weight_decay, optimizer_type, momentum,
                    max_epochs, base_lr, lr_epoch_boundaries, lr_decay_factor, linear_final_lr,
                    warmup_epochs, warmup_method, warmup_factor,
                    ckpt_iter_period, tb_iter_period,
                    output_dir, tb_dir, save_weights,
                    device='cuda', weight_decay_bias=0, bias_lr_factor=2, init_weights=None, val_epoch_period=-1, grad_accum_iters=1,
                            deps=None,
                            se_reduce_scale=0):
    print('----------------- show lr schedule --------------')
    print('base_lr:', base_lr)
    print('max_epochs:', max_epochs)
    print('lr_epochs:', lr_epoch_boundaries)
    print('lr_decay:', lr_decay_factor)
    print('linear_final_lr:', linear_final_lr)
    print('-------------------------------------------------')

    return BaseConfigByEpoch(network_type=network_type,dataset_name=dataset_name,dataset_subset=dataset_subset,global_batch_size=global_batch_size,num_node=num_node, device=device,
                      weight_decay=weight_decay,weight_decay_bias=weight_decay_bias,optimizer_type=optimizer_type,momentum=momentum,bias_lr_factor=bias_lr_factor,
                      max_epochs=max_epochs, base_lr=base_lr, lr_epoch_boundaries=lr_epoch_boundaries,lr_decay_factor=lr_decay_factor, linear_final_lr=linear_final_lr,
                             warmup_epochs=warmup_epochs,warmup_method=warmup_method,warmup_factor=warmup_factor,
                      ckpt_iter_period=int(ckpt_iter_period),tb_iter_period=int(tb_iter_period),
                      output_dir=output_dir, tb_dir=tb_dir,
                      init_weights=init_weights, save_weights=save_weights,
                             val_epoch_period=val_epoch_period, grad_accum_iters=grad_accum_iters, deps=deps, se_reduce_scale=se_reduce_scale)

def get_baseconfig_for_test(network_type, dataset_subset, global_batch_size, init_weights, device='cuda', deps=None, se_reduce_scale=0):
    return BaseConfigByEpoch(network_type=network_type, dataset_name=get_dataset_name_by_model_name(network_type),
                             dataset_subset=dataset_subset, global_batch_size=global_batch_size, num_node=1, device=device,
                             weight_decay=None, weight_decay_bias=None, optimizer_type=None, momentum=None, bias_lr_factor=None,
                             max_epochs=None, base_lr=None, lr_epoch_boundaries=None, lr_decay_factor=None, linear_final_lr=None,
                             warmup_epochs=None, warmup_method=None, warmup_factor=None, ckpt_iter_period=None,
                             tb_iter_period=None, output_dir=None, tb_dir=None, init_weights=init_weights,
                             save_weights=None, val_epoch_period=None, grad_accum_iters=None, deps=deps, se_reduce_scale=se_reduce_scale)