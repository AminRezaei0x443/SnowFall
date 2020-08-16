from snowfall.manager.execution_manager import ExecutionManager
from snowfall.nn import Circuit, Network
import torch.nn as nn

from snowfall.optimizers.adam import adam

net = Network()
net += nn.Linear(12, 36)
net += nn.Linear(36, 24)
net += nn.Linear(36, 4)

manager = ExecutionManager(use_gpu=True)


def viewer(x, y):
    return x.view(3, 4), y


def accuracy(pred, gt):
    pred = pred.argmax(dim=1, keepdim=True)
    correct = pred.eq(gt.view_as(pred)).sum()
    return correct.float()/pred.shape[0]


c = Circuit(manager)
c += net
c *= adam()
c -= nn.MSELoss()
c %= accuracy
c /= viewer
c.develop(train_data, val_data)
