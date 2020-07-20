OVERALL_EVAL_RECORD_FILE = 'overall_eval_records.txt'
from collections import namedtuple
import numpy as np

LRSchedule = namedtuple('LRSchedule', ['base_lr', 'max_epochs', 'lr_epoch_boundaries', 'lr_decay_factor',
                                       'linear_final_lr', 'cosine_minimum'])

VGG_ORIGIN_DEPS = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
CFQK_ORIGIN_DEPS = np.array([32, 32, 64], dtype=np.int32)

MI1_ORIGIN_DEPS = np.array([32,
                   32, 64,
                   64, 128,
                   128, 128,
                   128, 256,
                   256, 256,
                   256, 512,
                   512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
                   512, 1024,
                   1024, 1024])
MI1_PACESETTERS = list(range(0, 27, 2))
MI1_FOLLOW_DICT = {(i+1):i for i in range(0, 26, 2)}
MI1_NON_GROUPWISE_LAYERS = MI1_PACESETTERS
MI1_SUCC_STRATEGY = {i:(i+1) for i in range(27)}


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


RC56_LAYER_INPUT_SIZE = {i:32 for i in range(0, 21)}
RC56_LAYER_INPUT_SIZE.update({i:16 for i in range(21, 40)})
RC56_LAYER_INPUT_SIZE.update({i:8 for i in range(40, 57)})

MI1_LAYER_INPUT_SIZE = {0:224}
MI1_LAYER_INPUT_SIZE.update({i:112 for i in [1, 2, 3]})
MI1_LAYER_INPUT_SIZE.update({i:56 for i in [4, 5, 6, 7]})
MI1_LAYER_INPUT_SIZE.update({i:28 for i in [8, 9, 10, 11]})
MI1_LAYER_INPUT_SIZE.update({i:14 for i in range(12, 24)})
MI1_LAYER_INPUT_SIZE.update({i:7 for i in [24, 25, 26]})

#   3, 4, 6, 3
RESNET50_PW_LAYERS = [1,3,4,6,7,9,
                      10, 11,13,14,16,17,19,20,22,
                      23, 24,26,27,29,30,32,33,35,36,38,39,41,
                      42, 43,45,46,48,49,51]
for i in range(len(RESNET50_PW_LAYERS)):
    RESNET50_PW_LAYERS[i] += 1
RESNET50_PW_LAYERS.insert(0, 1)

RESNET50_INTERNAL_KERNEL_IDXES = [2,3,5,6,8,9,
                                  12,13,15,16,18,19,21,22,
                                  25,26,28,29,31,32,34,35, 37,38,40,41,
                                  44,45,47,48,50,51]


import numpy as np

##################### general Resnet on CIFAR-10
def rc_origin_deps_flattened(n):
    assert n in [3, 9, 12, 18, 27, 200]
    filters_in_each_stage = n * 2 + 1
    stage1 = [16] * filters_in_each_stage
    stage2 = [32] * filters_in_each_stage
    stage3 = [64] * filters_in_each_stage
    return np.array(stage1 + stage2 + stage3, dtype=np.int)


def rc_convert_flattened_deps(flattened):
    filters_in_each_stage = len(flattened) / 3
    n = int((filters_in_each_stage - 1) // 2)
    assert n in [3, 9, 12, 18, 27, 200]
    pacesetters = rc_pacesetter_idxes(n)
    result = [flattened[0]]
    for ps in pacesetters:
        assert flattened[ps] == flattened[ps+2]
        stage_deps = []
        for i in range(n):
            stage_deps.append([flattened[ps + 1 + 2 * i], flattened[ps + 2 + 2 * i]])
        result.append(stage_deps)
    return result



def rc_pacesetter_idxes(n):
    assert n in [3, 9, 12, 18, 27, 200]
    filters_in_each_stage = n * 2 + 1
    pacesetters = [0, int(filters_in_each_stage), int(2 * filters_in_each_stage)]
    return pacesetters

def rc_internal_layers(n):
    assert n in [3, 9, 12, 18, 27, 200]
    pacesetters = rc_pacesetter_idxes(n)
    result = []
    for ps in pacesetters:
        for i in range(n):
            result.append(ps + 1 + 2 * i)
    return result

def rc_all_survey_layers(n):
    return rc_pacesetter_idxes(n) + rc_internal_layers(n)

def rc_all_cov_layers(n):
    return range(0, 6*n+3)

def rc_pacesetter_dict(n):
    assert n in [3, 9, 12, 18, 27, 200]
    pacesetters = rc_pacesetter_idxes(n)
    result = {}
    for ps in pacesetters:
        for i in range(0, n+1):
            result[ps + 2 * i] = ps
    return result

def rc_succeeding_strategy(n):
    assert n in [3, 9, 12, 18, 27, 200]
    internal_layers = rc_internal_layers(n)
    result = {i : (i+1) for i in internal_layers}
    result[0] = 1
    follow_dic = rc_pacesetter_dict(n)
    pacesetters = rc_pacesetter_idxes(n)
    layer_before_pacesetters = [i-1 for i in pacesetters]
    for i in follow_dic.keys():
        if i in layer_before_pacesetters:
            result[i] = [i+1, i+2]
        elif i not in pacesetters:
            result[i] = i + 1
    return result

def rc_fc_layer_idx(n):
    assert n in [9, 12, 18, 27, 200]
    return 6*n+3

def rc_stage_to_pacesetter_idx(n):
    ps_ids = rc_pacesetter_idxes(n)
    return {2:ps_ids[0], 3:ps_ids[1], 4:ps_ids[2]}

def rc_flattened_deps_by_stage(rc_n, stage2, stage3, stage4):
    result = rc_origin_deps_flattened(rc_n)
    stage2_ids = (result == 16)
    stage3_ids = (result == 32)
    stage4_ids = (result == 64)
    result[stage2_ids] = stage2
    result[stage3_ids] = stage3
    result[stage4_ids] = stage4
    return result


def convert_flattened_resnet50_deps(deps):
    assert len(deps) == 53
    assert deps[1] == deps[4] and deps[11] == deps[14] and deps[24] == deps[27] and deps[43] == deps[46]
    d = [deps[0]]
    tmp = []
    for i in range(3):
        tmp.append([deps[2 + i * 3], deps[3 + i * 3], deps[4 + i * 3]])
    d.append(tmp)
    tmp = []
    for i in range(4):
        tmp.append([deps[12 + i * 3], deps[13 + i * 3], deps[14 + i * 3]])
    d.append(tmp)
    tmp = []
    for i in range(6):
        tmp.append([deps[25 + i * 3], deps[26 + i * 3], deps[27 + i * 3]])
    d.append(tmp)
    tmp = []
    for i in range(3):
        tmp.append([deps[44 + i * 3], deps[45 + i * 3], deps[46 + i * 3]])
    d.append(tmp)
    return d


RESNET50_ORIGIN_DEPS=[64,[[64,64,256]]*3,
                       [[128,128,512]]*4,
                       [[256, 256, 1024]]*6,
                       [[512, 512, 2048]]*3]
RESNET50_ORIGIN_DEPS_FLATTENED = [64,256,64,64,256,64,64,256,64,64,256,512,128,128,512,128,128,512,128,128,512,128,128,512,
                                  1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,
                                  2048,512, 512, 2048,512, 512, 2048,512, 512, 2048]
RESNET50_ALL_CONV_LAYERS = range(0, len(RESNET50_ORIGIN_DEPS_FLATTENED))



RESNET50_PACESETTER_IDXES = [1, 11, 24, 43]
RESNET50_ALL_SURVEY_LAYERS = [0] + RESNET50_INTERNAL_KERNEL_IDXES + RESNET50_PACESETTER_IDXES
RESNET50_FOLLOW_DICT = {1:1, 4:1, 7:1, 10:1, 11:11, 14:11, 17:11, 20:11, 23:11, 24:24, 27:24, 30:24, 33:24, 36:24, 39:24, 42:24, 43:43, 46:43, 49:43, 52:43}
# RESNET50_FOLLOWER_DICT = {1:[1,4,7,10], 11:[11,14,17,20,23], 24:[24,27,30,33,36,39,42], 43:[43,46,49,52]}
RESNET50_succeeding_STRATEGY = {i : (i+1) for i in RESNET50_INTERNAL_KERNEL_IDXES}
RESNET50_succeeding_STRATEGY[0] = [1,2]
idxes_before_pacesetters = [i-1 for i in RESNET50_PACESETTER_IDXES]
for i in RESNET50_FOLLOW_DICT.keys():
    if i not in RESNET50_PACESETTER_IDXES:
        if i in idxes_before_pacesetters:
            RESNET50_succeeding_STRATEGY[i] = [i+1, i+2]
        else:
            RESNET50_succeeding_STRATEGY[i] = i+1

resnet_n_to_num_blocks = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3)
}

def resnet_bottleneck_origin_deps_converted(res_n):
    num_blocks = resnet_n_to_num_blocks[res_n]
    return [64,[[64,64,256]]*num_blocks[0],
                       [[128,128,512]]*num_blocks[1],
                       [[256, 256, 1024]]*num_blocks[2],
                       [[512, 512, 2048]]*num_blocks[3]]

def _resnet_bottlenck_first_internal_layer_idx_of_stage(num_blocks):
    return [2, 3+num_blocks[0]*3, 4+(num_blocks[0]+num_blocks[1])*3, 5+(num_blocks[0]+num_blocks[1]+num_blocks[2])*3]


def convert_resnet_bottleneck_deps(deps):
    # print('converting: ', deps) #TODO
    assert len(deps) in [53, 104, 155]
    res_n = len(deps) - 3
    # print('converting the flattened deps of resnet-{}'.format(res_n))
    num_blocks = resnet_n_to_num_blocks[res_n]
    #   the idx of the first layer of the stage (not the proj layer)
    start_layer_idx_of_stage = _resnet_bottlenck_first_internal_layer_idx_of_stage(num_blocks)
    d = [deps[0]]
    for stage_idx in range(4):
        tmp = []
        assert deps[start_layer_idx_of_stage[stage_idx] - 1] == deps[start_layer_idx_of_stage[stage_idx] + 2]  # check the proj layer deps
        for i in range(num_blocks[stage_idx]):
            tmp.append([deps[start_layer_idx_of_stage[stage_idx] + i * 3],
                        deps[start_layer_idx_of_stage[stage_idx] + 1 + i * 3], deps[start_layer_idx_of_stage[stage_idx] + 2 + i * 3]])
        d.append(tmp)
    # print('converting completed')
    return d

def resnet_bottleneck_origin_deps_flattened(res_n):
    origin_deps_converted = resnet_bottleneck_origin_deps_converted(res_n)
    flattened = [origin_deps_converted[0]]
    for stage_idx in range(4):
        flattened.append(origin_deps_converted[stage_idx+1][0][2])
        for block in origin_deps_converted[stage_idx+1]:
            flattened += block
    return flattened

def resnet_bottleneck_internal_kernel_indices(res_n):
    internals = []
    num_blocks = resnet_n_to_num_blocks[res_n]
    start_layer_idx_of_stage = _resnet_bottlenck_first_internal_layer_idx_of_stage(num_blocks)
    for stage_idx in range(4):
        for i in range(num_blocks[stage_idx]):
            internals.append(start_layer_idx_of_stage[stage_idx] + i * 3)
            internals.append(start_layer_idx_of_stage[stage_idx] + 1 + i * 3)
    return internals

def resnet_bottleneck_33_kernel_indices(res_n):
    internals = []
    num_blocks = resnet_n_to_num_blocks[res_n]
    start_layer_idx_of_stage = _resnet_bottlenck_first_internal_layer_idx_of_stage(num_blocks)
    for stage_idx in range(4):
        for i in range(num_blocks[stage_idx]):
            internals.append(start_layer_idx_of_stage[stage_idx] + 1 + i * 3)
    return internals

def resnet_bottleneck_pacesetter_indices(res_n):
    num_blocks = resnet_n_to_num_blocks[res_n]
    start_layer_idx_of_stage = _resnet_bottlenck_first_internal_layer_idx_of_stage(num_blocks)
    return [i-1 for i in start_layer_idx_of_stage]

def resnet_bottleneck_flattened_deps_shrink_by_stage(res_n, shrink_ratio, only_internals=True):
    result_deps = resnet_bottleneck_origin_deps_flattened(res_n=res_n)
    bottleneck_indices = resnet_bottleneck_pacesetter_indices(res_n)
    internals = resnet_bottleneck_internal_kernel_indices(res_n)
    for i in range(len(result_deps)):
        if only_internals and i not in internals:
            continue
        if i >= bottleneck_indices[3]:
            stage_order = 3
        elif i >= bottleneck_indices[2]:
            stage_order = 2
        elif i >= bottleneck_indices[1]:
            stage_order = 1
        elif i >= bottleneck_indices[0]:
            stage_order = 0
        else:
            stage_order = -1
        if stage_order >= 0:
            result_deps[i] = np.ceil(shrink_ratio[stage_order] * result_deps[i])
    result_deps =np.asarray(result_deps, dtype=np.int32)
    print('resnet {} deps shrinked by stage_ratio {} is {}'.format(res_n, shrink_ratio, result_deps))
    return result_deps


def resnet_bottleneck_follow_dict(res_n):
    num_blocks = resnet_n_to_num_blocks[res_n]
    pacesetters = resnet_bottleneck_pacesetter_indices(res_n)
    follow_dict = {}
    for stage_idx in range(4):
        for i in range(num_blocks[stage_idx] + 1):
            follow_dict[pacesetters[stage_idx] + 3 * i] = pacesetters[stage_idx]
    return follow_dict

def resnet_bottleneck_succeeding_strategy(res_n):
    internals = resnet_bottleneck_internal_kernel_indices(res_n)
    pacesetters = resnet_bottleneck_pacesetter_indices(res_n)
    follow_dict = resnet_bottleneck_follow_dict(res_n)
    result = {i : (i+1) for i in internals}
    result[0] = [1,2]
    layers_before_pacesetters = [i - 1 for i in pacesetters]
    for i in follow_dict.keys():
        if i not in pacesetters:
            if i in layers_before_pacesetters:
                result[i] = [i + 1, i + 2]
            else:
                result[i] = i + 1
    return result

def wrn_convert_flattened_deps(flattened):
    assert len(flattened) in [16, 28, 40]
    n = int((len(flattened) - 4) // 6)
    assert n in [2, 4, 6]
    pacesetters = wrn_pacesetter_idxes(n)
    result = [flattened[0]]
    for ps in pacesetters:
        assert flattened[ps] == flattened[ps+2]
        stage_deps = []
        for i in range(n):
            stage_deps.append([flattened[ps + 1 + 2 * i], flattened[ps + 2 + 2 * i]])
        result.append(stage_deps)
    return result