# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

from loader.data_loader import CIFAR_10
import torch
import numpy as np

if __name__ == "__main__":
    array = [[1,2], [3, 4]]
    size = len(array)
    cifar = CIFAR_10(array, size)
    print(cifar[0])