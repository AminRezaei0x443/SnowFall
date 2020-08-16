import torch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from functools import reduce
from torch.optim.optimizer import Optimizer
from snowfall.optimizers import Optimizer as Opt

# noinspection PyArgumentList
from snowfall.optimizers import LambdaOptimizer


class HessianFreeOptimizer(Optimizer):
    def __init__(self, params,
                 lr=1, damping=0.5, delta_decay=0.95,
                 conjugate_grad_iters=100, generalized_gauss_newton_matrix=True,
                 log=False):
        super().__init__(params, {
            "alpha": lr,
            "damping": damping,
            "delta_decay": delta_decay,
            "conjugate_grad_iters": conjugate_grad_iters,
            "generalized_gauss_newton_matrix": generalized_gauss_newton_matrix,
            "log": log
        })
        if len(self.param_groups) != 1:
            raise ValueError("Parameter Groups Are Not Supported!")
        self._params = self.param_groups[0]['params']

    # noinspection PyMethodOverriding
    def step(self, closure, PC=None):
        """
        Performs an optimization step.
        """
        assert len(self.param_groups) == 1, "Parameter Groups Are Not Supported!"

        group = self.param_groups[0]
        alpha = group['alpha']
        delta_decay = group['delta_decay']
        conjugate_grad_iters = group['conjugate_grad_iters']
        damping = group['damping']
        generalized_gauss_newton_matrix = group['generalized_gauss_newton_matrix']
        log = group['log']

        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        oLoss, output = closure()
        current_evals = 1
        state['func_evals'] += 1

        # Gather current parameters and respective gradients
        flat_params = parameters_to_vector(self._params)
        flat_grad = self._flat_grad()

        A = (lambda x: self._Gv(oLoss, output, x, damping)) if generalized_gauss_newton_matrix \
            else (lambda x: self._Hv(flat_grad, x, damping))

        if PC is not None:
            pc_inv = PC()

            # Preconditioner recipe
            if pc_inv.dim() == 1:
                m = (pc_inv + damping) ** (-0.85)

                def M(x):
                    return m * x
            else:
                m = torch.inverse(pc_inv + damping * torch.eye(*pc_inv.shape))

                def M(x):
                    return m @ x
        else:
            M = None

        b = flat_grad.detach()

        # Initializing Conjugate-Gradient
        init_delta = torch.zeros_like(flat_params) if state.get('init_delta') is None \
            else delta_decay * state.get('init_delta')
        # Epsilon for type (Numeral System)
        eps = torch.finfo(b.dtype).eps
        # Conjugate-Gradient
        deltas, Ms = self._CG(A=A, b=b.neg(), x0=init_delta,
                              M=M, max_iter=conjugate_grad_iters,
                              tol=1e1 * eps, eps=eps, martens=True)
        # Update parameters
        delta = state['init_delta'] = deltas[-1]
        M = Ms[-1]
        vector_to_parameters(flat_params + delta, self._params)
        nLoss = closure()[0]
        current_evals += 1
        state['func_evals'] += 1
        # Conjugate-Gradient backtracking
        if log:
            print("Loss before Conjugate Gradient: {}".format(float(oLoss)))
            print("Loss before Backtracking: {}".format(float(nLoss)))

        for (d, m) in zip(reversed(deltas[:-1][::2]), reversed(Ms[:-1][::2])):
            vector_to_parameters(flat_params + d, self._params)
            loss_prev = closure()[0]
            if float(loss_prev) > float(nLoss):
                break
            delta = d
            M = m
            nLoss = loss_prev

        if log:
            print("Loss after Backtracking:  {}".format(float(nLoss)))

        # The Levenberg-Marquardt Heuristic
        reduction_ratio = (float(nLoss) - float(oLoss)) / M if M != 0 else 1

        if reduction_ratio < 0.25:
            group['damping'] *= 3 / 2
        elif reduction_ratio > 0.75:
            group['damping'] *= 2 / 3
        if reduction_ratio < 0:
            group['init_delta'] = 0

        # Line Searching
        beta = 0.8
        c = 1e-2
        min_improv = min(c * torch.dot(b, delta), 0)
        for _ in range(60):
            if float(nLoss) <= float(oLoss) + alpha * min_improv:
                break
            alpha *= beta
            vector_to_parameters(flat_params + alpha * delta, self._params)
            nLoss = closure()[0]
        else:  # No good update found
            alpha = 0.0
            nLoss = oLoss

        # Update the parameters (this time fo real)
        vector_to_parameters(flat_params + alpha * delta, self._params)

        if log:
            print("Loss after Line Searching:  {0} (lr: {1:.3f})".format(float(nLoss), alpha))
            print("Tikhonov damping: {0:.3f} (reduction ratio: {1:.3f})".format(group['damping'], reduction_ratio),
                  end='\n\n')

        return nLoss

    def _flat_grad(self):
        views = list()
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.contiguous().view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _CG(self, A, b, x0, M=None, max_iter=50, tol=1.2e-6, eps=1.2e-7, martens=False):
        """
            Minimize Residual Error Of Linear System Using Conjugate Gradient
        """
        x = [x0]
        r = A(x[0]) - b
        if M is not None:
            y = M(r)
            p = -y
        else:
            p = -r
        res_i_norm = r @ r
        if martens:
            m = [0.5 * (r - b) @ x0]
        for i in range(max_iter):
            Ap = A(p)
            alpha = res_i_norm / ((p @ Ap) + eps)
            x.append(x[i] + alpha * p)
            r = r + alpha * Ap
            if M is not None:
                y = M(r)
                res_ip1_norm = y @ r
            else:
                res_ip1_norm = r @ r
            beta = res_ip1_norm / (res_i_norm + eps)
            res_i_norm = res_ip1_norm
            # Martens' Relative Progress stopping condition
            if martens:
                m.append(0.5 * A(x[i + 1]) @ x[i + 1] - b @ x[i + 1])
                k = max(10, int(i / 10))
                if i > k:
                    stop = (m[i] - m[i - k]) / (m[i] + eps)
                    if stop < 1e-4:
                        break
            if res_i_norm < tol or torch.isnan(res_i_norm):
                break
            if M is not None:
                p = - y + beta * p
            else:
                p = - r + beta * p

        return (x, m) if martens else (x, None)

    def _R_operator(self, y, x, v, create_graph=False):
        """
        Computes the product (dy_i/dx_j) v_j: R-operator
        """
        if isinstance(y, tuple):
            ws = [torch.zeros_like(y_i, requires_grad=True) for y_i in y]
        else:
            ws = torch.zeros_like(y, requires_grad=True)

        jacobian = torch.autograd.grad(
            y, x, grad_outputs=ws, create_graph=True)

        Jv = torch.autograd.grad(parameters_to_vector(
            jacobian), ws, grad_outputs=v, create_graph=create_graph)

        return parameters_to_vector(Jv)

    def _Hv(self, gradient, vec, damping):
        """
        Computes the Hessian vector product.
        """
        Hv = self._R_operator(gradient, self._params, vec)
        # Tikhonov damping
        return Hv.detach() + damping * vec

    def _Gv(self, loss, output, vec, damping):
        """
        Computes the generalized Gauss-Newton vector product.
        """
        Jv = self._R_operator(output, self._params, vec)
        gradient = torch.autograd.grad(loss, output, create_graph=True)
        HJv = self._R_operator(gradient, output, Jv)
        JHJv = torch.autograd.grad(output, self._params, grad_outputs=HJv.reshape_as(output), retain_graph=True)
        # Tikhonov damping
        return parameters_to_vector(JHJv).detach() + damping * vec


def empirical_fisher_diagonal(model, xb, yb, criterion):
    grads = list()
    for (x, y) in zip(xb, yb):
        fi = criterion(model(x), y)
        grads.append(torch.autograd.grad(fi, model.parameters(), retain_graph=False))
    vec = torch.cat([(torch.stack(p) ** 2).mean(0).detach().flatten() for p in zip(*grads)])
    return vec


def hessian_free(lr=1, damping=0.5, delta_decay=0.95,
                 conjugate_grad_iters=100, generalized_gauss_newton_matrix=True,
                 log=False):
    return LambdaOptimizer(lambda model: HessianFreeOptimizer(model.parameters(), lr, damping, delta_decay,
                                                              conjugate_grad_iters, generalized_gauss_newton_matrix,
                                                              log))


Opt.register_opt("hessian_free", hessian_free)
