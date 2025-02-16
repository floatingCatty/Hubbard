import numpy as np

class PDIIS(object):
    def __init__(self, p0, a: float=0.05, n: int=6, k: int=3, **kwargs):
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

        self.R = [None for _ in range(n)]
        self.F = [None for _ in range(n)]

        self._p = p0
        self.nparam = p0.size
        self.pshape = p0.shape

        self.a = a
        self.n = n
        self.k = k

        self._iter = 0

    def update(self, p: np.ndarray):

        f = p - self._p

        if self._iter > 0:
            self.F[(self._iter-1) % self.n] = f - self.f

        if not (self._iter+1) % self.k:
            F_ = np.stack([t for t in self.F if t is not None]).reshape(-1, self.nparam)
            R_ = np.stack([t for t in self.R if t is not None]).reshape(-1, self.nparam)
            p_ = self._p + self.a*f - ((R_.T+self.a*F_.T)@np.linalg.inv(F_ @ F_.T) @ F_ @ f.flatten()).reshape(*self.pshape)
        else:
            p_ = self._p + self.a * f

        self.R[self._iter % self.n] = p_ - self._p

        self.f = f.copy()
        self._p = p_.copy()

        self._iter += 1
        
        return p_

    def reset(self, p0):
        self._iter = 0
        self.R = [None for _ in range(self.n)]
        self.F = [None for _ in range(self.n)]
        self._p = p0

        return True
    
    
class Linear(object):
    def __init__(self, p0, a: float=0.05, **kwargs):
        """Linear mixing

        Args:
            p0 (_type_): the initial point
            a (float, optional): the mixing beta value, or step size. Defaults to 0.05.
            n (int, optional): the size of the storage of history to compute the pesuedo hessian matrix. Defaults to 6.
        Returns:
            p _type_: the stable point
        """

        self._p = p0
        self.a = a

    def update(self, p: np.ndarray):

        new_p = (1-self.a) * self._p + self.a * p
        self._p = new_p.copy()

        return new_p

    def reset(self, p0):
        self._p = p0

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

