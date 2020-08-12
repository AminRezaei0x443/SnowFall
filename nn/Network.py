from typing import Any

import torch.nn as nn
from torch.nn.modules.module import T_co

from err.unsupported_arg import UnsupportedArg


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.modules = []

    def add_layer(self, module, name=None):
        if name is None:
            name = len(self.modules) + 1
        self.modules.append((name, module))
        self.add_module(name, module)


    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        return self.inner(*input)

    def __iadd__(self, other):
        name = None
        if other is tuple:
            if len(other) == 1:
                module = other[0]
            elif len(other) == 2:
                module = other[0]
                name = other[1]
            else:
                raise UnsupportedArg("Unsupported Signature, Sig: (module, [name])")
        else:
            module = other
        if not isinstance(module, nn.Module):
            raise UnsupportedArg("nn.Module object is required")
        self.add_layer(module, name)
