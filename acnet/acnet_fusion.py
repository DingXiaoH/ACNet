from utils.misc import read_hdf5, save_hdf5
import numpy as np

SQUARE_KERNEL_KEYWORD = 'square_conv.weight'

def _fuse_kernel(kernel, gamma, std):
    b_gamma = np.reshape(gamma, (kernel.shape[0], 1, 1, 1))
    b_gamma = np.tile(b_gamma, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
    b_std = np.reshape(std, (kernel.shape[0], 1, 1, 1))
    b_std = np.tile(b_std, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
    return kernel * b_gamma / b_std

def _add_to_square_kernel(square_kernel, asym_kernel):
    asym_h = asym_kernel.shape[2]
    asym_w = asym_kernel.shape[3]
    square_h = square_kernel.shape[2]
    square_w = square_kernel.shape[3]
    square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                                        square_w // 2 - asym_w // 2 : square_w // 2 - asym_w // 2 + asym_w] += asym_kernel


def convert_acnet_weights(train_weights, deploy_weights, eps):
    train_dict = read_hdf5(train_weights)
    print(train_dict.keys())
    deploy_dict = {}
    square_conv_var_names = [name for name in train_dict.keys() if SQUARE_KERNEL_KEYWORD in name]
    for square_name in square_conv_var_names:
        square_kernel = train_dict[square_name]
        square_mean = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.running_mean')]
        square_std = np.sqrt(train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.running_var')] + eps)
        square_gamma = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.weight')]
        square_beta = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.bias')]

        ver_kernel = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_conv.weight')]
        ver_mean = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.running_mean')]
        ver_std = np.sqrt(train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.running_var')] + eps)
        ver_gamma = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.weight')]
        ver_beta = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.bias')]

        hor_kernel = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_conv.weight')]
        hor_mean = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.running_mean')]
        hor_std = np.sqrt(train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.running_var')] + eps)
        hor_gamma = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.weight')]
        hor_beta = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.bias')]

        fused_bias = square_beta + ver_beta + hor_beta - square_mean * square_gamma / square_std \
                     - ver_mean * ver_gamma / ver_std - hor_mean * hor_gamma / hor_std
        fused_kernel = _fuse_kernel(square_kernel, square_gamma, square_std)
        _add_to_square_kernel(fused_kernel, _fuse_kernel(ver_kernel, ver_gamma, ver_std))
        _add_to_square_kernel(fused_kernel, _fuse_kernel(hor_kernel, hor_gamma, hor_std))

        deploy_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.weight')] = fused_kernel
        deploy_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.bias')] = fused_bias

    for k, v in train_dict.items():
        if 'hor_' not in k and 'ver_' not in k and 'square_' not in k:
            deploy_dict[k] = v
    save_hdf5(deploy_dict, deploy_weights)




