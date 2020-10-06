# coding=utf-8

import pickle
import os
import numpy as np
import torch

class DataLoader(object):
    def __init__(self, batch_size):
        super(DataLoader).__init__()
        self.batch_size = batch_size
        self.datas = None

    def __getitem__(self, item):
        if isinstance(self.datas, np.ndarray) or isinstance(self.datas, torch.Tensor) \
                or isinstance(self.datas, list):
            return self.datas[item]
        else:
            raise TypeError('data type not support')


class CIFAR_10(DataLoader):
    def __init__(self, path_or_list, batch_size):
        super(CIFAR_10).__init__(batch_size)
        if isinstance(path_or_list, str):
            #TODO: unused
            pass
        else:
            self.datas = path_or_list

    