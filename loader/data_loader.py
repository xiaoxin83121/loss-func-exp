# coding=utf-8

import pickle
import os
import numpy as np
import torch

class DataLoader(object):
    def __init__(self, size):
        super(DataLoader, self).__init__()
        self.size = size
        self.datas = None

    def __getitem__(self, item):
        if isinstance(self.datas, torch.Tensor):
            return self.datas[item]
        else:
            raise TypeError('data type not support')


class CIFAR_10(DataLoader):
    """
    :param:
    path_or_list: subject of dataset, now only support input numpy or tensor array
    """
    def __init__(self, path_or_list, size):
        super(CIFAR_10, self).__init__(size)
        if isinstance(path_or_list, torch.Tensor):
            self.datas = path_or_list
        elif isinstance(path_or_list, np.ndarray) or isinstance(path_or_list, list):
            self.datas = torch.tensor(path_or_list)
        else:
            raise TypeError('data type not support')




    