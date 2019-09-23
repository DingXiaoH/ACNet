OVERALL_EVAL_RECORD_FILE = 'overall_eval_records.txt'
from collections import namedtuple

LRSchedule = namedtuple('LRSchedule', ['base_lr', 'max_epochs', 'lr_epoch_boundaries', 'lr_decay_factor'])

import numpy as np


def parse_usual_lr_schedule(try_arg):
    if 'lrs1' in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=500, lr_epoch_boundaries=[100, 200, 300, 400], lr_decay_factor=0.1)
    elif 'lrs2' in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=360, lr_epoch_boundaries=[90, 180, 240, 300], lr_decay_factor=0.2)
    elif 'lrs' in try_arg:
        raise ValueError('Unsupported lrs config.')
    else:
        lrs = None
    return lrs



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