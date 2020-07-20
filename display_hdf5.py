from utils.misc import read_hdf5
import sys
import numpy as np

di = read_hdf5(sys.argv[1])
num_kernel_params = 0

conv_kernel_cnt = 0
matrix_param_cnt = 0
vec_param_cnt = 0

bias_cnt = 0
beta_cnt = 0
gamma_cnt = 0
mu_cnt = 0
var_cnt = 0

for name, array in di.items():
    if array.ndim in [2, 4]:
        num_kernel_params += array.size

    if 'base_mask' in name:
        print(name, array)

    print(name, array.shape, np.mean(array), np.std(array),
          ' positive {}, negative {}, zeros {}, near-zero {}'.format(np.sum(array > 0), np.sum(array < 0), np.sum(array == 0),
                                                                     np.sum(np.abs(array) <= 1e-5)))

    if array.ndim == 2:
        matrix_param_cnt += array.size
    elif array.ndim == 1:
        vec_param_cnt += array.size
    elif array.ndim == 4:
        conv_kernel_cnt += array.size
    if 'running_mean' in name or 'moving_mean' in name:
        mu_cnt += array.size
    elif 'running_var' in name or 'moving_var' in name:
        var_cnt += array.size
    elif ('weight' in name and 'bn' in name.lower()) or 'gamma' in name:
        gamma_cnt += array.size
    elif ('bias' in name and 'bn' in name.lower()) or 'beta' in name:
        beta_cnt += array.size
    elif 'bias' in name:
        bias_cnt += array.size
    elif 'spatial_mask' in name:
        print(array)
        print(np.sum(array))

print('number of kernel params: ', num_kernel_params)
print('vec {}, matrix {}, conv {}, total {}'.format(vec_param_cnt, matrix_param_cnt, conv_kernel_cnt,
                                                    vec_param_cnt + matrix_param_cnt + conv_kernel_cnt))
print('mu {}, var {}, gamma {}, beta {}, bias {}'.format(mu_cnt, var_cnt, gamma_cnt, beta_cnt, bias_cnt))
