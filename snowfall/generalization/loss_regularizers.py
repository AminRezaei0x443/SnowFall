import torch

from snowfall.losses import Loss


class LossRegularizer(Loss):
    def __init__(self, loss, lambda_):
        super().__init__()
        # set loss
        if isinstance(loss, str):
            if Loss.has(loss):
                loss = Loss.get(loss)
                self.loss = loss
            else:
                raise RuntimeError("Unregistered/Undefined Loss Function %s; Use object instead" % loss)
        else:
            self.loss = loss
        self.lambda_ = lambda_

    def forward(self, input, target):
        return self.loss(input, target)


class L1Regularize(LossRegularizer):
    def forward(self, input, target):
        model = self.get_prop("model")
        manager = self.get_prop("manager")
        l1_reg = torch.tensor(0., requires_grad=True).to(manager.current())
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.norm(param, 1)
        return self.loss(input, target) + self.lambda_ * l1_reg


class L2Regularize(LossRegularizer):
    def forward(self, input, target):
        model = self.get_prop("model")
        manager = self.get_prop("manager")
        l2_reg = torch.tensor(0., requires_grad=True).to(manager.current())
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param)
        return self.loss(input, target) + self.lambda_ * l2_reg


class GroupLassoRegularize(LossRegularizer):
    def forward(self, input, target):
        model = self.get_prop("model")
        manager = self.get_prop("manager")
        gl_reg = torch.tensor(0., requires_grad=True).to(manager.current())
        for name, param in model.named_parameters():
            if 'weight' in name:
                gl_reg = gl_reg + param.norm(2, dim=1).sum()
            if 'bias' in name:
                gl_reg = gl_reg + param.norm(2)
        return self.loss(input, target) + self.lambda_ * gl_reg


class SparseGroupLassoRegularize(LossRegularizer):
    def forward(self, input, target):
        model = self.get_prop("model")
        manager = self.get_prop("manager")
        gl_reg = torch.tensor(0., requires_grad=True).to(manager.current())
        l1_reg = torch.tensor(0., requires_grad=True).to(manager.current())
        for name, param in model.named_parameters():
            if 'weight' in name:
                gl_reg = gl_reg + param.norm(2, dim=1).sum()
                l1_reg = l1_reg + torch.norm(param, 1)
            if 'bias' in name:
                gl_reg = gl_reg + param.norm(2)
        return self.loss(input, target) + self.lambda_ * (gl_reg + l1_reg)
