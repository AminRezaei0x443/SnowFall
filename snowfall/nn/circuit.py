import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm

from snowfall.core import SnowObject
from snowfall.data import LambdasDataLoader
from snowfall.losses import Loss
from snowfall.nn import Network
from snowfall.optimizers import Optimizer


class Circuit(SnowObject):
    optimizer = None
    cost = None
    event_listeners = []
    preprocessors = []
    metrics = {}
    network: Network = None

    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.optimizer = None
        self.cost = None
        self.network = Network()
        self.event_listeners = []
        self.preprocessors = []
        self.metrics = {}

    def add_network(self, network):
        self.network.add_layers(network.modules)

    def add_listener(self, listener):
        self.event_listeners.append(listener)

    def add_metric(self, name, metric):
        self.metrics[name] = metric

    def add_preprocessor(self, preprocessor):
        if isinstance(preprocessor, tuple):
            pass
        else:
            self.preprocessors.append(preprocessor)

    def val_processors(self):
        f = filter(lambda x: True if not isinstance(x, tuple) else x[1], self.preprocessors)
        return list(map(lambda x: x if not isinstance(x, tuple) else x[0], f))

    def set_optimizer(self, optimizer, **kwargs):
        if isinstance(optimizer, str):
            if Optimizer.has(optimizer):
                optimizer = Optimizer.opt(optimizer)(**kwargs)
            else:
                raise RuntimeError("Unregistered/Undefined Optimizer %s; Use object instead" % optimizer)
        elif isinstance(optimizer, Optimizer):
            pass
        else:
            raise RuntimeError("Unsupported optimizer; Use Optimizer instance or key instead")
        self.optimizer = optimizer.build(self.network)

    def set_loss_func(self, loss):
        if isinstance(loss, str):
            if Loss.has(loss):
                loss = Loss.get(loss)
                self.cost = loss
            else:
                raise RuntimeError("Unregistered/Undefined Loss Function %s; Use object instead" % loss)
        elif isinstance(loss, Loss):
            self.cost = loss
            self.cost.add_prop("model", self.network)
            self.cost.add_prop("manager", self.manager)
        elif isinstance(loss, nn.Module):
            self.cost = loss
        else:
            raise RuntimeError("Unsupported Loss; Use Loss/nn.Module instance or key instead")

    def __iadd__(self, item):
        if isinstance(item, Network):
            self.add_network(item)
        else:
            super().__iadd__(item)
        return self

    def __imul__(self, optimizer):
        self.set_optimizer(optimizer)
        return self

    def __isub__(self, loss):
        self.set_loss_func(loss)
        return self

    def __ipow__(self, listener):
        self.add_listener(listener)
        return self

    def __imod__(self, metric):
        if isinstance(metric, list):
            for m in metric:
                if len(m) != 2:
                    raise RuntimeError("Values must be tuples in form of (name, metric)")
                self.add_metric(m[0], m[1])
        else:
            if len(metric) != 2:
                raise RuntimeError("Values must be tuples in form of (name, metric)")
            self.add_metric(metric[0], metric[1])
        return self

    def __ior__(self, preprocessor):
        if isinstance(preprocessor, list):
            for p in preprocessor:
                self.add_preprocessor(p)
        else:
            self.add_preprocessor(preprocessor)
        return self

    def develop(self, train_dataset, epochs, val_split=0.15, train_batch=32, val_batch=32, progress=True):
        total = len(train_dataset)
        val_count = int(val_split * total)
        train_set, val_set = random_split(train_dataset, (total - val_count, val_count))

        train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=val_batch, shuffle=True)
        train_loader = LambdasDataLoader(train_loader, self.preprocessors)
        val_loader = LambdasDataLoader(val_loader, self.val_processors())

        for listener in self.event_listeners:
            listener.add_prop("model", self)
            listener.add_prop("manager", self.manager)
            listener.add_prop("optimizer", self.optimizer)

        for i in range(epochs):
            # Training an epoch
            self.network.train()
            losses = {}
            metrics = {}
            for metric in self.metrics:
                metrics[metric] = {}
            it = enumerate(train_loader)
            it = tqdm(it, total=len(train_loader), position=0) if progress else it
            if progress:
                it.set_description('Epoch %d' % i)

            cmds = []
            for listener in self.event_listeners:
                r = listener.on_epoch_start(i)
                cmds.append(r)
            if any(cmds):
                return

            for batch_idx, (data, target) in it:
                data, target = data.to(self.manager.current()), target.to(self.manager.current())

                def fwd():
                    self.optimizer.zero_grad()
                    output = self(data)
                    loss = self.cost(output, target)
                    losses[batch_idx] = loss.item()
                    for metric in self.metrics:
                        metrics[metric][batch_idx] = self.metrics[metric](output, target).item()
                    loss.backward(create_graph=True)
                    return loss, output

                self.optimizer.step(closure=fwd)

                for listener in self.event_listeners:
                    listener.on_batch_processed(i, batch_idx, losses[batch_idx])

                if progress:
                    m_mean = {k: np.mean(list(metrics[k].values())) for k in self.metrics}
                    it.set_postfix(batch=batch_idx, loss=losses[batch_idx], mean_loss=np.mean(list(losses.values())),
                                   **m_mean)
            if progress:
                it.close()
            ls = list(losses.values())
            m_mean = {k: np.mean(list(metrics[k].values())) for k in self.metrics}
            m_all = {k + "_all": list(metrics[k].values()) for k in self.metrics}

            for listener in self.event_listeners:
                listener.on_epoch_end(i, ls, np.mean(ls), **m_mean, **m_all)
            # Validating an epoch
            self.network.eval()

            it = val_loader
            it = tqdm(it, total=len(val_loader), position=0) if progress else it
            if progress:
                it.set_description('Validating Epoch %d' % i)

            loss_val = []
            metrics_val = {}
            for metric in self.metrics:
                metrics_val[metric] = []
            with torch.no_grad():
                for data, target in it:
                    data, target = data.to(self.manager.current()), target.to(self.manager.current())
                    output = self(data)
                    loss = self.cost(output, target)
                    loss = loss.item()
                    loss_val.append(loss)
                    for metric in self.metrics:
                        metrics_val[metric].append(self.metrics[metric](output, target).item())
                    if progress:
                        m_mean = {k: np.mean(metrics_val[k]) for k in self.metrics}
                        it.set_postfix(loss=loss, mean_loss=np.mean(loss_val),
                                       **m_mean)
            m_mean = {k: np.mean(metrics_val[k]) for k in self.metrics}
            m_all = {k + "_all": metrics_val[k] for k in self.metrics}

            cmds = []
            for listener in self.event_listeners:
                r = listener.on_validated(i, np.mean(loss_val), **m_mean, **m_all)
                cmds.append(r)
            if any(cmds):
                return

            if progress:
                it.close()

    def evaluate(self, test_dataset, batch_size=32, progress=True):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_loader = LambdasDataLoader(test_loader, self.preprocessors)
        self.network.eval()

        it = test_loader
        it = tqdm(it, total=len(test_loader), position=0) if progress else it
        if progress:
            it.set_description("Evaluating")

        loss_val = []
        metrics_val = {}
        for metric in self.metrics:
            metrics_val[metric] = []
        outs = []
        with torch.no_grad():
            for data, target in it:
                data, target = data.to(self.manager.current()), target.to(self.manager.current())
                output = self(data)
                outs.append(output)
                loss = self.cost(output, target).item()
                loss_val.append(loss)
                for metric in self.metrics:
                    metrics_val[metric].append(self.metrics[metric](output, target).item())
                if progress:
                    m_mean = {k: np.mean(metrics_val[k]) for k in self.metrics}
                    it.set_postfix(loss=loss, mean_loss=np.mean(loss_val),
                                   **m_mean)
        m_mean = {k: np.mean(metrics_val[k]) for k in self.metrics}
        m_mean["loss"] = np.mean(loss_val)
        if progress:
            it.close()
        return torch.cat(outs), m_mean

    def process(self, proc_dataset, batch_size=32, progress=True):
        proc_loader = DataLoader(proc_dataset, batch_size=batch_size, shuffle=True)
        proc_loader = LambdasDataLoader(proc_loader, self.preprocessors)
        self.network.eval()

        it = proc_loader
        it = tqdm(it, total=len(proc_loader), position=0) if progress else it
        if progress:
            it.set_description("Processing")

        outs = []
        with torch.no_grad():
            for data in it:
                data = data.to(self.manager.current())
                output = self(data)
                outs.append(output)
        if progress:
            it.close()
        return torch.cat(outs)

    def __call__(self, x):
        return self.network(x)
