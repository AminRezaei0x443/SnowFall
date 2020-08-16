from typing import Any

import torch.nn as nn

from snowfall.core import SnowObject


class Network(nn.Module, SnowObject):
    def __init__(self):
        super().__init__()
        self.modules = []

    def add_layer(self, layer):
        name = None
        if isinstance(layer, tuple):
            if len(layer) == 1:
                module = layer[0]
            elif len(layer) == 2:
                module = layer[0]
                name = layer[1]
            else:
                raise RuntimeError("Unsupported Signature, Sig: (module, [name])")
        else:
            module = layer
        if not isinstance(module, nn.Module):
            raise RuntimeError("nn.Module object is required")
        if name is None:
            name = "l" + str(len(self.modules) + 1)
            self.reserve_prop(name)
        self.modules.append((module, name))
        self.add_module(name, module)

    def add_layers(self, layers):
        for layer in layers:
            self.add_layer(layer)

    def forward(self, x):
        res = x
        for module, name in self.modules:
            res = module(res)
        return res

    def __iadd__(self, other):
        if isinstance(other, list):
            self.add_layers(other)
        else:
            self.add_layer(other)
        return self
