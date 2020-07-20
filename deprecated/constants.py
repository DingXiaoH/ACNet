OVERALL_EVAL_RECORD_FILE = 'overall_eval_records.txt'
from collections import namedtuple

LRSchedule = namedtuple('LRSchedule', ['base_lr', 'max_epochs', 'lr_epoch_boundaries', 'lr_decay_factor',
                                       'linear_final_lr'])

import numpy as np


def parse_usual_lr_schedule(try_arg, keyword='lrs{}'):
    if keyword.format(1) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=500, lr_epoch_boundaries=[100, 200, 300, 400], lr_decay_factor=0.3,
                         linear_final_lr=None)
    elif keyword.format(2) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=500, lr_epoch_boundaries=[100, 200, 300, 400], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format(3) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=800, lr_epoch_boundaries=[200, 400, 600], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format(4) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=80, lr_epoch_boundaries=[20, 40, 60], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format(5) in try_arg:
        lrs = LRSchedule(base_lr=0.05, max_epochs=200, lr_epoch_boundaries=[50, 100, 150], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format(6) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=360, lr_epoch_boundaries=[90, 180, 240, 300], lr_decay_factor=0.2,
                         linear_final_lr=None)
    elif keyword.format(7) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=800, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=1e-4)
    elif keyword.format(8) in try_arg:  # may be enough for MobileNet v1 on CIFARs
        lrs = LRSchedule(base_lr=0.1, max_epochs=400, lr_epoch_boundaries=[100, 200, 300], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format(9) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=200, lr_epoch_boundaries=[50, 100, 150], lr_decay_factor=0.1,
                         linear_final_lr=None)

    elif keyword.format('A') in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=100, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=1e-5)
    elif keyword.format('B') in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=100, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=1e-6)
    elif keyword.format('C') in try_arg:
        lrs = LRSchedule(base_lr=0.2, max_epochs=125, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=0)
    elif keyword.format('D') in try_arg:
        lrs = LRSchedule(base_lr=0.001, max_epochs=20, lr_epoch_boundaries=[5, 10], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format('E') in try_arg:
        lrs = LRSchedule(base_lr=0.001, max_epochs=300, lr_epoch_boundaries=[100, 200], lr_decay_factor=0.1,
                         linear_final_lr=None)

    elif keyword.format('F') in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=120, lr_epoch_boundaries=[30, 60, 90, 110], lr_decay_factor=0.1,
                         linear_final_lr=None)
    #   for VGG and CFQKBN
    elif keyword.format('G') in try_arg:
        lrs = LRSchedule(base_lr=0.05, max_epochs=800, lr_epoch_boundaries=[200, 400, 600], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format('H') in try_arg:
        lrs = LRSchedule(base_lr=0.025, max_epochs=200, lr_epoch_boundaries=[50, 100, 150], lr_decay_factor=0.1,
                         linear_final_lr=None)

    elif keyword.format('X') in try_arg:
        lrs = LRSchedule(base_lr=0.2, max_epochs=6, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=0)

    elif keyword.replace('{}', '') in try_arg:
        raise ValueError('Unsupported lrs config.')
    else:
        lrs = None
    return lrs


VGG_ORIGIN_DEPS = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

CFQK_ORIGIN_DEPS = np.array([32, 32, 64], dtype=np.int32)





def wrn_origin_deps_flattened(n, k):
    assert n in [2, 4, 6]   # total_depth = 6n + 4
    filters_in_each_stage = n * 2 + 1
    stage0 = [16]
    stage1 = [16 * k] * filters_in_each_stage
    stage2 = [32 * k] * filters_in_each_stage
    stage3 = [64 * k] * filters_in_each_stage
    return np.array(stage0 + stage1 + stage2 + stage3)

def wrn_pacesetter_idxes(n):
    assert n in [2, 4, 6]
    filters_in_each_stage = n * 2 + 1
    pacesetters = [1, int(filters_in_each_stage)+1, int(2 * filters_in_each_stage)+1]   #[1, 10, 19] for WRN-28-x, for example
    return pacesetters

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


####################    WRN
WRN16_FOLLOW_DICT = {1:1, 3:1, 5:1, 6:6, 8:6, 10:6, 11:11, 13:11, 15:11}
WRN16_PACESETTER_IDS = [1, 6, 11]
WRN16_SUBSEQUENT_STRATEGY = {
    0:[1, 2],
    1:[4, 6, 7],
    2:3,
    4:5,
    6:[9, 11, 12],
    7:8,
    9:10,
    11:[14, 16],
    12:13,
    14:15
}
WRN16_INTERNAL_IDS = [2,4,7,9,12,14]

