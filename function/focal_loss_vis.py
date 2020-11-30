from __future__ import absolute_import
from matplotlib import pyplot as plt
import math
import numpy as np


def sigmoid(inp):
    return 1 / (1 + math.e ** (-inp))


def focal_loss(inp, gamma: float = 2.0):
    alpha = 0.5
    confidence = sigmoid(inp)
    return -alpha * (1 - confidence) ** gamma * math.log2(confidence)


def draw_focal_loss():
    gammas = [0, 0.5, 1, 2, 5]
    colors = ['r', 'b', 'g', 'c', 'm']
    x = np.arange(-5, 5, step=0.01)
    for i, gamma in enumerate(gammas):
        y = [focal_loss(inp, gamma) for inp in x]
        plt.plot(x, y, color=colors[i], label=str(gamma))
    plt.xlabel('x')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def focal_loss_star(inp, gamma=2, beta=0):
    alpha = 0.5
    con_star = sigmoid(gamma * inp + beta)
    return - alpha * math.log2(con_star) / gamma


def draw_focal_loss_star():
    paras = [[1, 0], [2, 0], [2, 1], [4, 0]]
    colors = ['r', 'b', 'g', 'c']
    x = np.arange(-5, 5, step=0.01)
    for i, para in enumerate(paras):
        y = [focal_loss_star(inp, gamma=para[0], beta=para[1]) for inp in x]
        plt.plot(x, y, color=colors[i], label="{} {}".format(para[0], para[1]))
    plt.xlabel('x')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def focal_loss_derivative(inp, gamma):
    p = sigmoid(inp)
    return (1-p)**gamma * (gamma * p * math.log2(p) + p - 1)


def draw_focal_loss_derivative():
    # gammas = [0, 0.5, 1, 2, 5]
    gammas = [0.5, 1, 2, 3, 5]
    colors = ['r', 'b', 'g', 'c', 'm']
    x = np.arange(-5, 5, step=0.01)
    for i, gamma in enumerate(gammas):
        y = [focal_loss_derivative(inp, gamma) for inp in x]
        plt.plot(x, y, color=colors[i], label=str(gamma))
    plt.xlabel('x')
    plt.ylabel('derivative')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # draw_focal_loss()
    # draw_focal_loss_star()
    draw_focal_loss_derivative()