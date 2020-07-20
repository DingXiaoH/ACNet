import torch
import time
from collections import OrderedDict
import os

def get_last_checkpoint(dir):
    'iter-200000.pth'
    target_ckpts = [t for t in os.listdir(dir) if '.pth' in t]
    if 'latest.pth' in target_ckpts:
        return os.path.join(dir, 'latest.pth')
    target_ckpts.sort(key=lambda x: int(x.replace('iter-', '').replace('.pth', '')))
    ckpt = os.path.join(dir, target_ckpts[-1])
    return ckpt

def load_model(model, model_file, logger):
    t_start = time.time()
    if isinstance(model_file, str):
        state_dict = torch.load(model_file, map_location='cpu')
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file

    state_dict = _align_and_update_loaded_state_dicts(
        model.state_dict(), state_dict)
    t_io_end = time.time()

    # if is_restore:
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = 'module.' + k
    #         new_state_dict[name] = v
    #     state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0 and logger is not None:
        logger.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0 and logger is not None:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    if logger is not None:
        logger.info(
            "Load model, Time usage:\n\tIO: {}, "
            "initialize parameters: {}".format(
                t_io_end - t_start, t_end - t_io_end))

    return model


def _align_and_update_loaded_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have
    prefixes appended to each of its keys, for example due to an extra
    level of nesting that the original pre-trained weights from ImageNet
    won't contain. For example, model.state_dict() might return
    backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys
    if there is one that is a suffix of the current weight name,
    and use it if that's the case. If multiple matches exist,
    take the one with longest size of the corresponding name. For example,
    for the same model as before, the pretrained weight file can contain
    both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    aligned_loaded_state_dict = loaded_state_dict.copy()

    # get a matrix of string matches, where each (i, j) entry
    # correspond to the size of the loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in
        loaded_keys]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys))
    max_match_size, idxs = match_matrix.max(1)
    idxs[max_match_size == 0] = -1

    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        aligned_loaded_state_dict[key] = \
            aligned_loaded_state_dict.pop(key_old)
    del loaded_state_dict
    return aligned_loaded_state_dict
