import numpy as np


class mixing(object):
    def __init__(self, p0, a, **kwargs):
        self.state = {"p0": p0, "a": a, **kwargs}

    def update(self):
        pass

    def reset():
        pass
    
    @classmethod
    def from_state(cls, state):
        mixer = cls(**state)
        return mixer

class PDIIS(mixing):
    def __init__(self, p0, a: float=0.05, n: int=6, k: int=3, **kwargs):
        super(PDIIS, self).__init__(p0=p0, a=a, n=n, k=k, **kwargs)
        """The periodic pully mixing from https://doi.org/10.1016/j.cplett.2016.01.033.

        Args:
            p0 (_type_): the initial point
            a (float, optional): the mixing beta value, or step size. Defaults to 0.05.
            n (int, optional): the size of the storage of history to compute the pesuedo hessian matrix. Defaults to 6.
            k (int, optional): the period of conducting pully mixing. The algorithm will conduct pully mixing every k iterations. Defaults to 3.
            tol (_type_, optional): the absolute err tolerance. Defaults to 1e-6.
            reltol (_type_, optional): the relative err tolerance. Defaults to 1e-3.

        Returns:
            p _type_: the stable point
        """

        # static
        # static variables must be provided when initialize the class, or can be infered from the initialized variables
        self.nparam = self.state["p0"].size
        self.pshape = self.state["p0"].shape
        self.a = a
        self.n = n
        self.k = k


        # dynamic
        # dynamic variables can be provided during initialization. If not, it need to be initialized.
        self.state["R"] = kwargs.get("R", [None for _ in range(n)])
        self.state["F"] = kwargs.get("F", [None for _ in range(n)])
        self.state["_iter"] = kwargs.get("_iter", 0)
        self.state["_p"] = kwargs.get("_p", self.state["p0"])
        self.state["_f"] = kwargs.get("_f", 0.)

        # self._iter = 0
        # self.R = [None for _ in range(n)]
        # self.F = [None for _ in range(n)]

        # self._p = p0
        # self.nparam = p0.size
        # self.pshape = p0.shape

    def update(self, p: np.ndarray):
        """
            R: dev history of new and old updated p
            p: incoming p
            p_: new updated p
            _p: old updated p
            f: dev of incoming p and old updated p
            _f: old f
            F: dev of new and old _f
        """

        f = p - self.state["_p"]

        if self.state["_iter"] > 0:
            self.state["F"][(self.state["_iter"]-1) % self.n] = f - self.state["_f"]

        if not (self.state["_iter"]+1) % self.k and self.state["_iter"] != 0:
            F_ = np.stack([t for t in self.state["F"] if t is not None]).reshape(-1, self.nparam)
            R_ = np.stack([t for t in self.state["R"] if t is not None]).reshape(-1, self.nparam)

            # handling numerical instability when F_ @ F_.T.conj() is degenerate
            distmat = F_ @ F_.T.conj()
            eigs = np.linalg.eigvalsh(distmat)
            if eigs[-1] / eigs[0] > 1000:
                # idx = (self.state["_iter"]-1) % self.n
                # # pop the most similar one in F_ and R_
                # cF = self.state["R"][idx]
                # cF = cF / np.linalg.norm(cF)
                # errs = []
                # for i in range(self.n):
                #     if i != idx and self.state["R"][i] is not None:
                #         err = (cF.conj() * self.state["R"][i]).sum() / np.linalg.norm(self.state["R"][i])
                #         errs.append(err)
                #     else:
                #         errs.append(0.)

                # self.state["F"][idx] = None
                # self.state["R"][idx] = None
                
                # idx = np.argmax(errs)

                # just throw away all except for the last one
                for i in range(self.n):
                    self.state["F"][i] = None
                    self.state["R"][i] = None
                
                p_ = self.state["_p"] + self.a * f
            else:
                print("cond #: ", eigs[-1] / eigs[0], eigs[-1], eigs[0])
                p_ = self.state["_p"] + self.a * f - ((R_.T.conj()+self.a*F_.T.conj()) @ np.linalg.solve(F_ @ F_.T.conj(), F_) @ f.flatten()).reshape(*self.pshape)
        else:
            p_ = self.state["_p"] + self.a * f

        self.state["R"][self.state["_iter"] % self.n] = p_ - self.state["_p"]

        self.state["_f"] = f.copy()
        self.state["_p"] = p_.copy()

        self.state["_iter"] += 1
        
        return p_

    def reset(self, p0):
        self.state["_iter"] = 0
        self.state["R"] = [None for _ in range(self.n)]
        self.state["F"] = [None for _ in range(self.n)]
        self.state["_p"] = p0
        self.state["_f"] = 0.

        return True
    
    
class Linear(mixing):
    def __init__(self, p0, a: float=0.05, **kwargs):
        super(Linear, self).__init__(p0=p0, a=a, **kwargs)
        """Linear mixing

        Args:
            p0 (_type_): the initial point
            a (float, optional): the mixing beta value, or step size. Defaults to 0.05.
            n (int, optional): the size of the storage of history to compute the pesuedo hessian matrix. Defaults to 6.
        Returns:
            p _type_: the stable point
        """

        self.state["_p"] = kwargs.get("_p", self.state["p0"])
        # self._p = p0
        self.a = a

    def update(self, p: np.ndarray):

        new_p = (1-self.a) * self.state["_p"] + self.a * p
        self.state["_p"] = new_p.copy()

        return new_p

    def reset(self, p0):
        self.state["_p"] = p0

        return True


# def PDIIS(fn, p0, a=0.05, n=6, maxIter=100, k=3, err=1e-6, relerr=1e-3, display=50, **kwargs):
#     """The periodic pully mixing from https://doi.org/10.1016/j.cplett.2016.01.033.

#     Args:
#         fn (function): the iterative functions
#         p0 (_type_): the initial point
#         a (float, optional): the mixing beta value, or step size. Defaults to 0.05.
#         n (int, optional): the size of the storage of history to compute the pesuedo hessian matrix. Defaults to 6.
#         maxIter (int, optional): the maximum iteration. Defaults to 100.
#         k (int, optional): the period of conducting pully mixing. The algorithm will conduct pully mixing every k iterations. Defaults to 3.
#         err (_type_, optional): the absolute err tolerance. Defaults to 1e-6.
#         relerr (_type_, optional): the relative err tolerance. Defaults to 1e-3.

#     Returns:
#         p _type_: the stable point
#     """
#     i = 0
#     f = fn(p0, **kwargs) - p0
#     p = p0
#     R = [None for _ in range(n)]
#     F = [None for _ in range(n)]
#     # print("SCF iter 0 abs err {0} | rel err {1}: ".format( 
#     #         f.abs().max().detach().numpy(), 
#     #         (f.abs() / p.abs()).max().detach().numpy())
#     #         )
#     while (f.abs().max() > err or (f.abs() / (p.abs()+1e-10)).max() > relerr) and i < maxIter:
#         if not (i+1) % k:
#             F_ = torch.stack([t for t in F if t != None])
#             R_ = torch.stack([t for t in R if t != None])
#             p_ = p + a*f - (R_.T+a*F_.T)@(F_ @ F_.T).inverse() @ F_ @ f
#         else:
#             p_ = p + a * f

#         f_ = fn(p_, **kwargs) - p_
#         F[i % n] = f_ - f
#         R[i % n] = p_ - p

#         p = p_.clone()
#         f = f_.clone()
#         i += 1

#         if i % display == 0:
#             print("Current: {0} with err {1} and rel_err {2}..".format(i, f.abs().max(), (f.abs() / (p.abs()+1e-10)).max()))

#         # print("SCF iter {0} abs err {1} | rel err {2}: ".format(
#         #     i, 
#         #     f.abs().max().detach().numpy(), 
#         #     (f.abs() / p.abs()).max().detach().numpy())
#         #     )


#     if i == maxIter:
#         print("Not Converged very well at {0} with err {1} and rel_err {2}.".format(i, f.abs().max(), (f.abs() / (p.abs()+1e-10)).max()))
#     else:
#         print("Converged very well at {0} with err {1} and rel_err {2}..".format(i, f.abs().max(), (f.abs() / (p.abs()+1e-10)).max()))


#     return p

