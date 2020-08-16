from torch.optim.optimizer import Optimizer as TOpt


class Optimizer:
    optimizers = {}

    def __init__(self):
        pass

    def build(self, model) -> TOpt:
        pass

    @staticmethod
    def register_opt(name, opt):
        Optimizer.optimizers[name] = opt

    @staticmethod
    def opt(name):
        return Optimizer.optimizers[name]

    @staticmethod
    def has(name):
        return name in Optimizer.optimizers


class LambdaOptimizer(Optimizer):
    def __init__(self, func):
        super().__init__()
        self.f = func

    def build(self, model) -> TOpt:
        return self.f(model)
