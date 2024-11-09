import torch
from torch.optim import Optimizer, Adam, SGD
import torch_optimizer as optim
from functorch import make_functional, jacfwd, jacrev, vmap
from torch.optim.optimizer import required
from gGA.data import _keys
import math

def newton(model, loss, data, lr):
    model_fn, params = make_functional(model)

    def fn(*params):
        return loss(model_fn(params, data))

    loss_value = fn(*params)
    jac = torch.autograd.grad(loss_value, params, create_graph=True)
    jac_flatten = [j.flatten() for j in jac]
    hessian = torch.autograd.functional.hessian(func=fn, inputs=params)
    inv00, inv01, inv11 = block_inversion(
        hessian[0][1].reshape(jac_flatten[0].shape[0], jac_flatten[1].shape[0]), 
        hessian[1][1].reshape(jac_flatten[1].shape[0], jac_flatten[1].shape[0])
        )

    delta_p = []
    j1 = jac_flatten[0]
    j2 = jac_flatten[1]
    delta_p.append(inv00@j1+inv01@j2)
    delta_p.append(inv01.T@j1+inv11@j2)

    for i, p in enumerate(model.parameters()):
        delta = delta_p[i].view_as(p)
        with torch.no_grad():
            p.add_(delta * lr)
    
    return loss_value.detach()


def block_inversion(A, B):
    """
    inverse a block matrix M = [
        [0,   A],
        [A.T, B]
    ]
    """
    Binv = B.inverse()
    ABinv = torch.linalg.solve(B.T, A.T).T
    inv00 = -(ABinv @ A.T).inverse()
    inv01 = -inv00 @ ABinv
    inv11 = Binv - ABinv.T @ inv01

    return inv00, inv01, inv11


class AugmentedLagrange(object):
    def __init__(self, model, lr, lossfn, lagrangefn, criterion, sch_gamma=0.5, sch_size=80, patience=50, delta=1e-3, delta_la=1e-3):
        super(AugmentedLagrange, self).__init__()
        self.model = model
        self.lossfn = lossfn
        self.lagrangefn = lagrangefn
        # self.lagrangeD = torch.randn((model.totalnorb*2*model.naux, model.totalnorb*2*model.naux), device=model.device, dtype=model.dtype).abs()
        # self.lagrangeR = torch.randn(
        #     (len(model.atomic_number), model.totalnorb*2*model.naux, model.totalnorb*2), 
        #     device=model.device, dtype=model.dtype).abs()
        # self.lagrangeR = self.lagrangeR - self.lagrangefn.bdiag(self.lagrangefn.bdiag(self.lagrangeR))

        self.criterion = criterion
        self.muD = 1.0
        # self.muR = 1.0
        # self.muNorm = 100.0
        self.delta = delta
        self.delta_la = delta_la
        self.sch_size = sch_size
        self.lr = lr
        self.patience = patience
        self.bdiag = torch.vmap(torch.diag)

        self.optimizer = Adam(list(model.interaction.parameters())+list(model.kinetic.parameters()), lr=lr)
        self.optimizer_la = Adam(model.lagrangian, lr=lr, maximize=True)
        # self.optimizer_laR = Adam([model.lagrangianR], lr=lr, maximize=True)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=sch_size, gamma=sch_gamma)

        self.count = 0
        self.countR = 0
        self.best_loss = 1e9
        self.best_lossR = 1e9

    def step(self, data):
        data = self.model(data)
        devDensity = (data[_keys.REDUCED_DENSITY_MATRIX_KEY] - data[_keys.VARIATIONAL_DENSITY_KEY]).abs().max()
        loss = self.lossfn(data)
        # if devDensity < 2e-2:
        #     # R = data[_keys.R_MATRIX_KEY].clone()
        #     # R = R / R.norm(dim=1, keepdim=True)
        #     # RTR = torch.bmm(R.transpose(1,2), R)
        #     # Rdiag = self.bdiag(self.bdiag(RTR))
        #     # Roffdiag = RTR - Rdiag

        #     # R_loss = (self.model.lagrangianR * (Roffdiag**2)).sum()
        #     loss = data[_keys.INTERACTION_ENERGY_KEY] + data[_keys.TOTAL_ENERGY_KEY] # + R_loss
        # else:
        #     loss = data[_keys.TOTAL_ENERGY_KEY]
        # l1loss = self.lagrangefn(data, self.lagrangeD, self.lagrangeR)
        # l2loss = self.muD * ((l1loss[0]/(self.lagrangeD+1e-13))**2).mean() + self.muR * ((l1loss[1]/(self.lagrangeR+1e-13))**2).mean()
        # l1loss = l1loss[0].mean() + l1loss[1].mean()

        # loss = loss + l1loss + l2loss

        # l1loss = self.lagrangefn(data, self.model.lagrangian)
        # l2loss = self.muD * ((data[_keys.REDUCED_DENSITY_MATRIX_KEY] - data[_keys.VARIATIONAL_DENSITY_KEY])**2).sum()
        l1loss = (torch.block_diag(*self.model.lagrangian) * ((data[_keys.REDUCED_DENSITY_MATRIX_KEY] - data[_keys.VARIATIONAL_DENSITY_KEY])**2)).sum()
        # l2loss = self.muD * ((data[_keys.REDUCED_DENSITY_MATRIX_KEY] - data[_keys.VARIATIONAL_DENSITY_KEY])**2).sum()

        

        loss = loss + l1loss # + R_loss

        # doing the projection
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        
        

        if devDensity > 1e-4:
            self.optimizer_la.step()
        
        if devDensity > 5e-3:
            if devDensity < (self.best_loss*0.9):
                self.best_loss = devDensity
                self.count = 0
            elif self.count > self.patience:
                self.count = 0
                self.optimizer_la.param_groups[0]['lr'] = self.optimizer_la.param_groups[0]['lr'] * 1.2
            else:
                self.count += 1

        # if devDensity < 2e-2:
        #     devR = Roffdiag.abs().max()
        #     if devR > 1e-4:
        #         self.optimizer_laR.step()
            
        #     if devR > 5e-3:
        #         if devR < (self.best_lossR*0.9):
        #             self.best_lossR = devR
        #             self.countR = 0
        #         elif self.countR > self.patience:
        #             self.countR = 0
        #             self.optimizer_laR.param_groups[0]['lr'] = self.optimizer_laR.param_groups[0]['lr'] * 1.2
        #         else:
        #             self.count += 1

        # this step use grad
        ct = self.criterion(self.model)

        self.optimizer.zero_grad()
        self.optimizer_la.zero_grad()
        # self.optimizer_laR.zero_grad()

        # data = self.model(data)
        # la_loss = self.lagrangefn(data, self.model.lagrangian[:-1]).sum()
        # R = data[_keys.R_MATRIX_KEY].clone()
        # R = R / R.norm(dim=1, keepdim=True)
        # RTR = torch.bmm(R.transpose(1,2), R)
        # Rdiag = self.bdiag(self.bdiag(RTR))
        # Roffdiag = (RTR - Rdiag).abs()

        # R_loss = (self.model.lagrangian[-1] * Roffdiag).sum()
        # (la_loss+R_loss).backward()

        # self.optimizer_la.step()
        # self.muD *= (1 + ((data[_keys.REDUCED_DENSITY_MATRIX_KEY] - data[_keys.VARIATIONAL_DENSITY_KEY])**2).mean().sqrt().detach().item())
        # for p in self.model.parameters():
        #     p.data = p.data / p.data.norm()
        
        # self.lr_scheduler_la.step()

        

        # self.optimizer.zero_grad()
        # self.optimizer_la.zero_grad()

        # if ct < self.delta:
        #     # update mu and lagrange
        #     with torch.no_grad():
        #         data = self.model(data)

        #         # pnorms = []
        #         # for p in self.model.parameters():
        #         #     pnorms.append(p.norm()-1)
        #         # pnorms = torch.stack(pnorms)
        #         # laglossD, laglossR = self.lagrangefn(data, self.lagrangeD, self.lagrangeR)
        #         # laglossD = self.lagrangefn(data, self.model.lagrangian)
        #         # laglossD /= torch.block_diag(*self.model.lagrangian) + 1e-13
        #         # laglossR /= self.lagrangeR + 1e-13

        #         laglossD = data[_keys.REDUCED_DENSITY_MATRIX_KEY] - data[_keys.VARIATIONAL_DENSITY_KEY]

        #         if laglossD.abs().max() > self.delta_la:
                #     self.muD *= (1 + laglossD.detach().norm())
                #     # self.lagrangeD += 2 * self.muD * laglossD

                #     count = 0
                #     for p in self.model.lagrangian:
                #         n = p.data.shape[0]
                #         p.data = p.data + 2 * self.muD * laglossD[count:count+n, count:count+n]
                #         count += n

                #     # self.model.lagrangian += 2 * self.muD * laglossD
                    
                #     # reset the lr_scheduler
                #     self.optimizer.param_groups[0]['lr'] = self.lr
                
                # # if laglossR.abs().max() > self.delta_la:
                # #     self.muR *= (2 + laglossR.detach().norm())
                # #     self.lagrangeR += 2 * self.muR * laglossR
                    
                # #     # reset the lr_scheduler
                # #     self.optimizer.param_groups[0]['lr'] = self.lr
                
                # # if pnorms.abs().max() > self.delta_la:
                # #     self.muNorm *= (2 + pnorms.norm())
                    
                # #     # reset the lr_scheduler
                # #     self.optimizer.param_groups[0]['lr'] = self.lr
        
        return data, loss, ct
        


class AdaiV2(Optimizer):
    r"""Implements AdaiV2.
    It is a generalized variant of Adai based on
    `Adaptive Inertia: Disentangling the Effects of Adaptive Learning Rate and Momentum`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        betas (Tuple[float, float], optional): beta0 and beta2 (default: (0.1, 0.99))
        eps (float, optional): the inertia bound (default: 1e-03)
        weight_decay (float, optional): weight decay (default: 0)
        dampening (float, optional): dampening for momentum (default: 1.)
        decoupled (boolean, optional): decoupled weight decay (default: True)
    """

    def __init__(self, params, lr=required, betas=(0.1, 0.99), eps=1e-03,
                 weight_decay=0, dampening=1., decoupled=True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0]:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= dampening <=1:
            raise ValueError("Invalid weight_decay value: {}".format(dampening))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, dampening=dampening, decoupled=decoupled)
        super(AdaiV2, self).__init__(params, defaults)
    

    def __setstate__(self, state):
        super(AdaiV2, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('decoupled', True)
            
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        param_size = 0
        exp_avg_sq_hat_sum = 0.
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_size += p.numel()
                grad = p.grad.data
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Cumulative products of beta1
                    state['beta1_prod'] = torch.ones_like(p.data, memory_format=torch.preserve_format)
                    
                state['step'] += 1

                exp_avg_sq = state['exp_avg_sq']
                beta0, beta2 = group['betas']

                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0 and group['decoupled'] == False:
                    grad.add_(p.data, alpha=group['weight_decay'])
                elif group['weight_decay'] != 0 and group['decoupled'] == True:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                exp_avg_sq_hat_sum += exp_avg_sq.sum() / bias_correction2
                
        # Calculate the mean of all elements in exp_avg_sq_hat
        exp_avg_sq_hat_mean = exp_avg_sq_hat_sum / param_size

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                state = self.state[p]

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1_prod = state['beta1_prod']
                beta0, beta2 = group['betas']

                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                beta1 = (1. - (exp_avg_sq_hat / exp_avg_sq_hat_mean).pow(1. / (3 - 2 * group['dampening'])).mul(beta0)).clamp(0., 1 - group['eps'])
                beta3 = (1. - beta1).pow(group['dampening'])
                
                beta1_prod.mul_(beta1)
                bias_correction1 = 1 - beta1_prod
                
                exp_avg.mul_(beta1).addcmul_(beta3, grad)
                exp_avg_hat = exp_avg / bias_correction1 * math.pow(beta0, 1. - group['dampening'])

                p.data.add_(exp_avg_hat, alpha=-group['lr'])

        return loss