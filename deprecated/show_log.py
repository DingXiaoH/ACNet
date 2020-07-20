import glob
import re
import numpy as np

top1_pattern = re.compile('top1=(\-*\d+(?:\.\d+)?)')
top5_pattern = re.compile('top5=(\-*\d+(?:\.\d+)?)')
loss_pattern = re.compile('loss=(\-*\d+(?:\.\d+)?)')


def get_value_by_pattern(pattern, line):
    return float(re.findall(pattern, line)[0])

def parse_top1_top5_loss_from_log_line(log_line):
    top1 = get_value_by_pattern(top1_pattern, log_line)
    top5 = get_value_by_pattern(top5_pattern, log_line)
    loss = get_value_by_pattern(loss_pattern, log_line)
    return top1, top5, loss


root_dir = 'acnet_exps'
num_logs = 10

log_files = glob.glob('{}/*/log.txt'.format(root_dir))



for file_path in log_files:
    top1_list = []
    top5_list = []
    loss_list = []
    with open(file_path, 'r') as f:
        origin_lines = f.readlines()
        log_lines = [l for l in origin_lines if 'top1' in l]
        last_lines = log_lines[-num_logs:]
    for l in last_lines:
        top1, top5, loss = parse_top1_top5_loss_from_log_line(l)
        top1_list.append(top1)
        top5_list.append(top5)
        loss_list.append(loss)
    network_try_arg = file_path.split('/')[1].replace('_train', '')
    print('{}, \t top1={:.3f}, \t top5={:.3f}, \t loss={:.5f}, \t {} logs'.format(network_try_arg, np.mean(top1_list), np.mean(top5_list), np.mean(loss_list), len(top1_list)))





