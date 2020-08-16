from torch.optim import Adam

from .optimizer import LambdaOptimizer, Optimizer


def adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
         weight_decay=0, amsgrad=False):
    return LambdaOptimizer(lambda model: Adam(model.parameters(), lr, betas, eps, weight_decay, amsgrad))


Optimizer.register_opt("adam", adam)
