import os
import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from configs import data_rt


def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def generate_root(sv_rt):
    PATH = sv_rt
    dir = os.listdir(PATH)
    name_idx = 1
    while True:
        name = 'ex' + str(name_idx)
        filePath = PATH + '/' + name
        if os.path.exists(filePath):
            name_idx += 1
            continue
        PATH += '/' + name
        os.mkdir(PATH)
        return PATH


def plt_and_save(loss, train_acc, test_acc, lrs, ROOT):
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("epoch")
    ax1.plot(loss, label="train loss")
    ax1.set_xlabel("epoch")
    ax1.legend()
    fig1.savefig(ROOT + '/loss.png')
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel("epoch")
    ax2.plot(train_acc, label="train accuracy")
    ax2.plot(test_acc, label="test accuracy")
    ax2.legend()
    fig2.savefig(ROOT + '/acc.png')

    fig3, ax3 = plt.subplots()
    ax3.set_xlabel("epoch")
    ax3.plot(lrs, label="learning rate")
    ax3.legend()
    fig3.savefig(ROOT + '/lr.png')


def get_CIFAR10(bs=128, Resize=False):
    if Resize:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.Resize([224, 224])
             ])
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

    batch_size = bs

    trainset = torchvision.datasets.CIFAR10(root=data_rt, train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_rt, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader

# plt a line connecting two models
def plt_connection_line(loss_fn, theta1, theta2, alpha_s, alpha_e):
    alphas = np.linspace(alpha_s, alpha_e, 50)
    losses = [loss_fn(alpha * theta1 + (1 - alpha) * theta2) for alpha in alphas]
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.plot(alphas, losses, label="train loss")
    ax.legend()
    plt.show()