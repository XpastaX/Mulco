import os
import random
import torch
import numpy as np
import json
import time


def read_json(path):
    data = []
    with open(path, 'r', encoding='UTF-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def check_dir(path, creat=True):
    if not os.path.exists(path):
        if creat:
            os.makedirs(path)
            print('Folder %s has been created.' % path)
            return True
        else:
            print('Folder not found.')
            return False
    else:
        return True


def set_seed(seed):
    """
    :param seed:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def write_log(txt, path, prt=True):
    if prt:
        print(txt)
    with open(path, 'a') as file:
        file.writelines(txt)


def join_cate(cate1, cate2):
    return '-'.join((cate1, cate2))


def check_type(obj):
    from types import ModuleType
    import inspect
    for attribute in dir(obj):
        attribute_value = getattr(obj, attribute)
        # print(f'{attribute=}, {type(attribute_value)=}\n')
        if isinstance(attribute_value, ModuleType) or inspect.ismodule(attribute_value) or type(
                attribute_value) is type(inspect):
            print(attribute_value)
    print('')


class ProgressBar(object):
    '''
    custom progress bar
    '''

    def __init__(self, n_total, width=30, desc='Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.6f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')
