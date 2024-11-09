import torch
from torch.optim import LBFGS, Adam
from xitorch.linalg.solve import solve
from xitorch.grad.jachess import jac

class _SCF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fcn, x0, options, maxIter=100, err=1e-7, method='PDIIS', *params):
        # with torch.no_grad():
        #     x_ = fcn(x0, *params)

        x_ = fcn(x0, *params)

        if method == "default":
            it = 0
            old_x = x0
            while (x_-old_x).norm() > err and it < maxIter:
                it += 1
                old_x = x_
                x_ = fcn(x_, *params)

        elif method == 'GD':
            x_ = x_.detach().requires_grad_()
            temp_p = [p.detach() for p in params]
            it = 0
            loss = 1

            def new_fcn(x_):

                loss = (x_ - fcn(x_, *temp_p)).abs().sum()
                print(loss)
                return loss

            with torch.enable_grad():
                while it < maxIter and loss > err:
                    it += 1
                    loss = new_fcn(x_)
                    x_ = x_ - 1e-3 * torch.autograd.grad(loss, (x_,))[0]

        elif method == 'Adam':
            # x = torch.randn(200,1, dtype=torch.float64)
            # x = x / x.norm()
            # x_ = x_.unsqueeze(1) @ x.T
            x_ = x_.detach().requires_grad_()
            temp_p = [p.detach() for p in params]
            optim = Adam(params=[x_], lr=1e-3)
            def new_fcn(x_):
                loss = (x_ - fcn(x_, *temp_p)).norm()
                print(loss)
                return loss
            i = 0
            loss = 1
            with torch.enable_grad():
                while i < maxIter and loss > err:
                    optim.zero_grad()
                    loss = new_fcn(x_)
                    loss.backward()
                    optim.step()


        elif method == "PDIIS":
            with torch.no_grad():
                x_ = PDIIS(lambda x: fcn(x, *params), p0=x_, maxIter=maxIter, **options)

        elif method == 'LBFGS':
            x_ = x_.detach().requires_grad_()
            temp_p = [p.detach() for p in params]
            optim = LBFGS(params=[x_], lr=1e-2)

            def new_fcn():
                optim.zero_grad()
                loss = (x_ - fcn(x_, *temp_p)).norm()
                loss.backward()
                print(loss)
                return loss

            with torch.enable_grad():
                for i in range(maxIter):
                    optim.step(new_fcn)
                    print(x_)

        else:
            raise ValueError

        print("Convergence achieved !")
        x_ = x_ + 0j
        ctx.save_for_backward(x_, *params)
        ctx.fcn = fcn

        return x_

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ = ctx.saved_tensors[0].detach().requires_grad_()
        params = ctx.saved_tensors[1:]

        idx = [i for i in range(len(params)) if params[i].requires_grad]


        fcn = ctx.fcn
        def new_fcn(x, *params):
            return x - fcn(x, *params)

        with torch.enable_grad():
            grad = jac(fcn=new_fcn, params=(x_, *params), idxs=[0])[0]

        # pre = solve(grad.H, -grad_outputs.reshape(-1, 1))
        pre = solve(grad.H, -grad_outputs.reshape(-1, 1).type_as(x_))
        pre = pre.reshape(grad_outputs.shape)


        with torch.enable_grad():
            params_copy = [p.detach().requires_grad_() for p in params]
            yfcn = new_fcn(x_, *params_copy)

        grad = torch.autograd.grad(yfcn, [params_copy[i] for i in idx], grad_outputs=pre,
                                   create_graph=torch.is_grad_enabled(),
                                   allow_unused=True)
        grad_out = [None for _ in range(len(params))]
        for i in range(len(idx)):
            grad_out[idx[i]] = grad[i]


        return None, None, None, None, None, None, *grad_out


def PDIIS(fn, p0, a=0.05, n=6, maxIter=100, k=3, err=1e-6, relerr=1e-3, display=50, **kwargs):
    """The periodic pully mixing from https://doi.org/10.1016/j.cplett.2016.01.033.

    Args:
        fn (function): the iterative functions
        p0 (_type_): the initial point
        a (float, optional): the mixing beta value, or step size. Defaults to 0.05.
        n (int, optional): the size of the storage of history to compute the pesuedo hessian matrix. Defaults to 6.
        maxIter (int, optional): the maximum iteration. Defaults to 100.
        k (int, optional): the period of conducting pully mixing. The algorithm will conduct pully mixing every k iterations. Defaults to 3.
        err (_type_, optional): the absolute err tolerance. Defaults to 1e-6.
        relerr (_type_, optional): the relative err tolerance. Defaults to 1e-3.

    Returns:
        p _type_: the stable point
    """
    i = 0
    f = fn(p0, **kwargs) - p0
    p = p0
    R = [None for _ in range(n)]
    F = [None for _ in range(n)]
    # print("SCF iter 0 abs err {0} | rel err {1}: ".format( 
    #         f.abs().max().detach().numpy(), 
    #         (f.abs() / p.abs()).max().detach().numpy())
    #         )
    while (f.abs().max() > err or (f.abs() / (p.abs()+1e-10)).max() > relerr) and i < maxIter:
        if not (i+1) % k:
            F_ = torch.stack([t for t in F if t != None])
            R_ = torch.stack([t for t in R if t != None])
            p_ = p + a*f - (R_.T+a*F_.T)@(F_ @ F_.T).inverse() @ F_ @ f
        else:
            p_ = p + a * f

        f_ = fn(p_, **kwargs) - p_
        F[i % n] = f_ - f
        R[i % n] = p_ - p

        p = p_.clone()
        f = f_.clone()
        i += 1

        if i % display == 0:
            print("Current: {0} with err {1} and rel_err {2}..".format(i, f.abs().max(), (f.abs() / (p.abs()+1e-10)).max()))

        # print("SCF iter {0} abs err {1} | rel err {2}: ".format(
        #     i, 
        #     f.abs().max().detach().numpy(), 
        #     (f.abs() / p.abs()).max().detach().numpy())
        #     )


    if i == maxIter:
        print("Not Converged very well at {0} with err {1} and rel_err {2}.".format(i, f.abs().max(), (f.abs() / (p.abs()+1e-10)).max()))
    else:
        print("Converged very well at {0} with err {1} and rel_err {2}..".format(i, f.abs().max(), (f.abs() / (p.abs()+1e-10)).max()))


    return p

scf = _SCF.apply
