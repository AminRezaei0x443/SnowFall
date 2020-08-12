import torch
import torch.nn.functional as F


def loss(predicted, ground_truth):
    return F.nll_loss(predicted, ground_truth)


def loss_n(model, lambda_):
    def loss(predicted, ground_truth):
        return F.nll_loss(predicted, ground_truth)

    return loss


def loss_l1(model, lambda_):
    def loss(predicted, ground_truth):
        l1_reg = torch.tensor(0., requires_grad=True).cuda()
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_reg += torch.norm(param, 1)
        return F.nll_loss(predicted, ground_truth) + lambda_ * l1_reg

    return loss


def loss_l2(model, lambda_):
    def loss(predicted, ground_truth):
        l2_reg = torch.tensor(0., requires_grad=True).cuda()
        for param in model.parameters():
            l2_reg += torch.norm(param)
        return F.nll_loss(predicted, ground_truth) + lambda_ * l2_reg

    return loss


def loss_group_lasso(model, lambda_):
    def loss(predicted, ground_truth):
        gl_reg = torch.tensor(0., requires_grad=True).cuda()
        for name, param in model.named_parameters():
            if 'weight' in name:
                gl_reg += param.norm(2, dim=1).sum()
            if 'bias' in name:
                gl_reg += param.norm(2)
        return F.nll_loss(predicted, ground_truth) + lambda_ * gl_reg

    return loss


def loss_sparse_group_lasso(model, lambda_):
    def loss(predicted, ground_truth):
        gl_reg = torch.tensor(0., requires_grad=True).cuda()
        l1_reg = torch.tensor(0., requires_grad=True).cuda()

        for name, param in model.named_parameters():
            if 'weight' in name:
                gl_reg += param.norm(2, dim=1).sum()
                l1_reg += torch.norm(param, 1)
            if 'bias' in name:
                gl_reg += param.norm(2)
        return F.nll_loss(predicted, ground_truth) + lambda_ * (gl_reg + l1_reg)

    return loss
