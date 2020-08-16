from torch import nn

from snowfall.core import SnowObject


class Loss(nn.Module, SnowObject):
    losses = {}

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        pass

    @staticmethod
    def register_loss(name, opt):
        Loss.losses[name] = opt

    @staticmethod
    def get(name):
        return Loss.losses[name]

    @staticmethod
    def has(name):
        return name in Loss.losses
