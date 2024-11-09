from model import slater_kanamori
from gGA.nn.kinetics import Kinetics
import torch
import torch.nn as nn
import torch.linalg as LA
import numpy as np
from model.operators import get_anni, dm_up, dm_down
from utils import ksum
import matplotlib.pyplot as plt

class Gutzwiller(nn.Module):
    def __init__(self, norb, t, U, Up, J, Jp, l, R, kpoints, kspace, Nocc, delta_deg=1e-5):
        super(Gutzwiller, self).__init__()
        self.interaction = slater_kanamori(norb, U, Up, J, Jp, l)
        self.basis = LA.eigh(self.interaction)[1].T
        self.nbasis = self.basis.shape[0]
        self.kspace = kspace
        self.Nocc = Nocc
        self.norb = norb
        self.t = t
        self.nk = len(kpoints)
        self.delta_deg = delta_deg

        # precomputation
        self.M = torch.zeros(norb,2,norb,2,self.nbasis,self.nbasis)
        for a in range(norb):
            for b in range(norb):
                for s in range(2): # 0 for up, 1 for down
                    for s_ in range(2):
                        matrices_left = []
                        matrices_right = []
                        for a_ in range(norb):
                            if a_ == a:
                                matrices_left.append(get_anni(s=s).H.contiguous())
                            else:
                                matrices_left.append(torch.eye(4))
                        for b_ in range(norb):
                            if b_ == b:
                                matrices_right.append(get_anni(s=s_))
                            else:
                                matrices_right.append(torch.eye(4))
                        fa = matrices_left[0]
                        fb = matrices_right[0]
                        for a_ in range(1,norb):
                            fa = torch.kron(fa, matrices_left[a_])
                            fb = torch.kron(fb, matrices_right[a_])

                        self.M[a,s,b,s_] = torch.einsum("ai,ib,bc,cj,jd,da->ij", self.basis.T, self.basis, fa, self.basis.T, self.basis, fb)
        self.M = (self.M + self.M.transpose(-2,-1)) / 2 # orb, spin, orb, spin, basis, basis

        self.interaction = torch.einsum("ai,ib,bc,cj,ja->ij", self.basis.T, self.basis, self.interaction, self.basis.T, self.basis) + 1e-5 * torch.eye(self.nbasis)
        self.N = torch.zeros(norb,2,norb,2,self.nbasis,self.nbasis)
        for a in range(norb):
            for b in range(norb):
                for s in range(2): # 0 for up, 1 for down
                    for s_ in range(2):
                        matrices_left = []
                        matrices_right = []
                        for a_ in range(norb):
                            if a_ == a:
                                matrices_left.append(get_anni(s=s).H.contiguous())
                            else:
                                matrices_left.append(torch.eye(4))
                        for b_ in range(norb):
                            if b_ == b:
                                matrices_right.append(get_anni(s=s_))
                            else:
                                matrices_right.append(torch.eye(4))
                        fa = matrices_left[0]
                        fb = matrices_right[0]
                        for a_ in range(1,norb):
                            fa = torch.kron(fa, matrices_left[a_])
                            fb = torch.kron(fb, matrices_right[a_])
                        self.N[a,s,b,s_] = torch.einsum("ai,ib,bj,jc,cd,da->ij", self.basis.T, self.basis, self.basis.T, self.basis, fa, fb)
        self.N = (self.N + self.N.transpose(-2,-1)) / 2
        # doing basis transformation

        # self.F = torch.einsum("ai,ib,bj,ja->ij", self.basis.T, self.basis, self.basis.T, self.basis)

        self.kin = Kinetics(
            t=t,
            R=R,
            kpoints=kpoints,
            kspace=kspace
        )

        # initilize the parameters
        # self.lbd = torch.nn.Parameter(torch.randn(norb,2,norb,2))
        self.c = torch.nn.Parameter(torch.randn(self.nbasis))
        self.W = torch.nn.Parameter(torch.randn(len(kpoints), 2 * self.norb, self.Nocc*2))

        

    def forward(self, rU=True):

        # preprocessing the parameters
        C = LA.qr(A=self.W, mode="reduced").Q.transpose(1,2) # the wave vector [nkpoints, 2 * Nocc, 2*norb]
        c = self.c / self.c.norm() # the basis vector for phi_i

        # compute the estimated density

        # doing basis transformation to T
        var_density = torch.einsum("n, asbpnm, m->asbp", c, self.N, c) # norb, 2, norb, 2
        var_density = var_density.reshape(self.norb*2, self.norb*2)
        n = var_density.diag()
        M_ = self.M / ((n*(1-n))**0.5).reshape(1,1,self.norb,2,1,1)
        R = torch.einsum("n,m,asbpnm->asbp", c, c, M_) # norb, 2, norb, 2
        t_ = torch.einsum("kasef,ascd,efbp->kcdbp", self.t, R, R) # nkpoints/nei, norb, 2, norb, 2

        self.kin.update_t(t=t_)

        # compute the expectation value
        # T = self.kin.vHv(vec=C) / C.shape[0]
        T = self.kin.vHv(vec=C)
        # T, C = torch.linalg.eigh(self.kin.get_hamiltonian())
        # C = C.transpose(1,2)
        # sort the T to get first nk * Nocc's eigenvector
        efermi = T.flatten().sort().values[self.nk*self.Nocc]
        mask = T.lt(efermi-self.delta_deg) # nk, Nocc*2 The degeneracy might cause many problems here
        # random choose degenerated states in the fermi level
        nstates = mask.sum()
        if nstates < self.nk * self.Nocc:
            state_left = self.nk * self.Nocc - nstates
            mask_left = T.ge(efermi-self.delta_deg).flatten() * T.lt(efermi+self.delta_deg).flatten()
            ndegstates = mask_left.sum()
            # selected_index = np.random.choice(mask_left.eq(1).nonzero().flatten(), (ndegstates-state_left,), replace=False)
            # mask_left[selected_index] = 0
            # assert mask_left.sum() == state_left, "selected states: {}, left states: {}".format(mask_left.sum(), state_left)
            mask_left = mask_left.reshape(mask.shape)
        else:
            mask_left = None
        # T_with_lbd = self.kin.vHv(vec=real_C, lagrange=self.lbd).sum()
        C_m = C[mask]
        real_C2 = torch.einsum("ni, nj->nij", C_m, C_m).sum(dim=0) # nstates, norb*2, norb*2
        if mask_left is not None:
            C_ml = C[mask_left]
            real_C2 += (state_left/ndegstates) * torch.einsum("ni, nj->nij", C_ml, C_ml).sum(dim=0)
            T = (T[mask].sum() + (state_left/ndegstates) * T[mask_left].sum()) / self.nk
        else:
            T = T[mask].sum() / self.nk
        real_C2 /= self.nk
        # real_C2 = torch.stack([(C[i][mask[i]]**2).sum(dim=0) for i in range(self.nk)]) # nk, norb*2
        density_psi0 = real_C2.diag()

        

        # T = T.sum() / self.nk
        U = c @ self.interaction @ c
        # lbd_density = (self.lbd.reshape(var_density.shape) * var_density).sum()

        # penalty_offdiag_density = ((var_density - var_density.diag().diag()) ** 2).mean().sqrt()
        penalty_diag_density = (n.sum() - self.Nocc).abs()

        penalty_density = ((real_C2 - var_density)**2).mean().sqrt()

        # return T+U+30*penalty_diag_density-lbd_density
        if rU:
            return T+U+10*penalty_diag_density+10*penalty_density
        else:
            return T+10*penalty_diag_density+10*penalty_density

    def analysis(self):
        """
            To see whether this procedure converged, we need to check the error of the psi0 are eigenvector of T^G.
        """
        with torch.no_grad():
            # preprocessing the parameters
            C = LA.qr(A=self.W, mode="reduced").Q.transpose(1,2) # the wave vector [nkpoints, 2 * Nocc, 2*norb]
            c = self.c / self.c.norm() # the basis vector for phi_i

            # compute the estimated density

            # doing basis transformation to T
            var_density = torch.einsum("n, asbpnm, m->asbp", c, self.N, c) # norb, 2, norb, 2
            n = var_density.reshape(self.norb*2, self.norb*2).diag()
            M_ = self.M / ((n*(1-n))**0.5).reshape(1,1,self.norb,2,1,1)
            R = torch.einsum("n,m,asbpnm->asbp", c, c, M_) # norb, 2, norb, 2
            t_ = torch.einsum("kasbp,ascd,efbp->kcdef", self.t, R, R) # nkpoints/nei, norb, 2, norb, 2

            self.kin.update_t(t=t_)

            # compute the expectation value
            T = self.kin.vHv(vec=C)
            # T, C = torch.linalg.eigh(self.kin.get_hamiltonian())
            # C = C.transpose(1,2)
            # T = self.kin.vHv(vec=C)
            # sort the T to get first nk * Nocc's eigenvector
            efermi = T.flatten().sort().values[self.nk*self.Nocc]
            mask = T.lt(efermi-self.delta_deg) # nk, Nocc*2 The degeneracy might cause many problems here
            # random choose degenerated states in the fermi level
            nstates = mask.sum()
            if nstates < self.nk * self.Nocc:
                state_left = self.nk * self.Nocc - nstates
                mask_left = T.gt(efermi-self.delta_deg).flatten() * T.lt(efermi+self.delta_deg).flatten()
                ndegstates = mask_left.sum()
                # selected_index = np.random.choice(mask_left.eq(1).nonzero().flatten(), (ndegstates-state_left,), replace=False)
                # mask_left[selected_index] = 0
                # assert mask_left.sum() == state_left, "selected states: {}, left states: {}".format(mask_left.sum(), state_left)
                print("degenerated states: {}, required states: {}".format(ndegstates, state_left))
                mask_left = mask_left.reshape(mask.shape)
            else:
                mask_left = None

            C_m = C[mask]
            real_C2 = torch.einsum("ni, nj->nij", C_m, C_m).sum(dim=0) # nstates, norb*2, norb*2
            if mask_left is not None:
                C_ml = C[mask_left]
                real_C2 += (state_left/ndegstates) * torch.einsum("ni, nj->nij", C_ml, C_ml).sum(dim=0)
            real_C2 /= self.nk
            # real_C2 = torch.stack([(C[i][mask[i]]**2).sum(dim=0) for i in range(self.nk)]) # nk, norb*2
            # T_with_lbd = self.kin.vHv(vec=real_C, lagrange=self.lbd).sum()
            density_psi0 = real_C2.diag()
            if mask_left is not None:
                T = (T[mask].sum() + (state_left/ndegstates) * T[mask_left].sum()) / self.nk
            else:
                T = T[mask].sum() / self.nk
            U = c @ self.interaction @ c
            # lbd_density = (self.lbd * var_density).sum()

            var_density = var_density.reshape(self.norb*2, self.norb*2)
            penalty_offdiag_density = ((var_density - var_density.diag().diag()) ** 2).mean().sqrt()
            penalty_diag_density = (var_density.diag().sum() - self.Nocc).abs()

        # I want to output these term seperatly to better visualize the training process
        print(
            "penalty_density: ", ((n-density_psi0)**2).mean().sqrt().item(), 
            "penalty_diag_density: ", penalty_diag_density.item(),
            "T+U", T.item()+U.item(),
            "T: ", T.item(),
            "U: ", U.item(),
            "N: ", density_psi0.sum().item(),
            "efermi: ", efermi.item(),
            "\n"
            )
        # print("R: ", R.reshape(self.norb*2, self.norb*2))

        return True

    def get_density(self):
        with torch.no_grad():
            c = self.c / self.c.norm() # the basis vector for phi_i

            # compute the estimated density

            # doing basis transformation to T
            var_density = torch.einsum("n, asbpnm, m->asbp", c, self.N, c) # norb, 2, norb, 2
        
        return var_density

    def get_R(self):
        with torch.no_grad():
            c = self.c / self.c.norm() # the basis vector for phi_i

            # compute the estimated density
            # doing basis transformation to T
            var_density = torch.einsum("n, asbpnm, m->asbp", c, self.N, c) # norb, 2, norb, 2
            n = var_density.reshape(self.norb*2, self.norb*2).diag()
            M_ = self.M / ((n*(1-n))**0.5).reshape(1,1,self.norb,2,1,1)
            R = torch.einsum("n,m,asbpnm->asbp", c, c, M_) # norb, 2, norb, 2

        return R