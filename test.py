# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

# from loader.data_loader import CIFAR_10
import numpy as np
import json

if __name__ == "__main__":
    # array = [[1,2], [3, 4]]
    # size = len(array)
    # cifar = CIFAR_10(array, size)
    # print(cifar[0])
    pass

    # a = {'a': 10}
    # writer = open('bbox.json', 'w')
    # json.dump(a, writer)
    a = [0.01, 0.04]
    x = np.arange(0, 0.1, step=0.01)
    y = np.zeros_like(x)
    y[np.where(x==a[0])[0]] += 1
    print(y)