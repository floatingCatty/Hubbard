import torch
import torch.nn as nn
from typing import Tuple, Union, Dict
from gGA.model.interaction import slater_kanamori
from gGA.model.operators import generate_basis, generate_basis_minimized, states_to_indices, generate_product_basis
from gGA.model.operators import annis
from gGA.utils.safe_svd import safeSVD

class gGASingleOrb(nn.Module):
    def __init__(
            self, 
            norb, 
            naux:int=1, 
            Hint_params:dict={}, 
            device: Union[str, torch.device] = torch.device("cpu")
            ):
        
        super(gGASingleOrb, self).__init__()
        self.dtype = torch.get_default_dtype()
        self.device = device
        self.norb = norb
        self.naux = naux

        self.phy_spinorb = self.norb*2
        self.aux_spinorb = self.naux*self.norb*2
        
        assert naux >= 1

        # here we decompose the basis set, since the constraint are that the physical+auxiliary basis set is half filled.
        # we have basis for the physical system with occupation number [0, 1, ..., 2*norb]
        # and have basis for the auxiliary system with occupation number [(naux+1)*norb, (naux+1)*norb-1, ..., (naux-1)*norb]

        # we compute the basis number of each
        self.basis = generate_basis(norb=(naux+1)*norb, nocc=(naux+1)*norb)
        self.basis = torch.tensor(self.basis, device=self.device, dtype=torch.int32)
        self.basis = torch.cat([self.basis[:,:norb], 3-self.basis[:,norb:]], dim=1)
        # self.basis = generate_product_basis(norb_A=norb, norb_B=naux*norb, nocc_A=norb, nocc_B=naux*norb)
        # we use a coo index to map the basis's states to the SB amplitude tensor
        self.indices = torch.stack([states_to_indices(self.basis[:, :norb]), states_to_indices(self.basis[:, norb:])])
        self.indices = self.indices.to(self.device)
        self.size = (4**norb, 4**(naux*norb))

        # setup U
        self.Hint = slater_kanamori(norb=norb, **Hint_params).to_sparse()
        self.Hint = self.Hint.to(self.device)

        self.SBamp = torch.randn(len(self.basis), dtype=self.dtype, device=self.device)
        # self.SBamp = self.SBamp / torch.norm(self.SBamp)
        self.SBamp = nn.Parameter(self.SBamp)
        # self.pstart = torch.randn(len(self.basis), dtype=self.dtype, device=self.device)
        # self.pstart = self.pstart / (torch.norm(self.pstart)+1e-7)

        # self.SBamp = torch.rand(len(self.basis)-1, dtype=self.dtype, device=self.device)
        self.SBamp = nn.Parameter(self.SBamp)
    
    def forward(self):
        # compute U, n, and solve for R.
        
        # first normalize the embedding states
        # sb = torch.tanh(self.SBamp)
        sb = self.SBamp / torch.norm(self.SBamp)
        # sb = sb / torch.norm(sb)
        # sb = torch.sparse_coo_tensor(self.indices, sb, self.size).coalesce()
        # vnorm = self.SBamp.norm()
        # sb = torch.cat([2*self.SBamp, vnorm.unsqueeze(0)**2-1]) / (vnorm**2+1)
        sb = torch.sparse_coo_tensor(self.indices, sb, self.size).coalesce()

        

        sbbs = torch.sparse.mm(sb, sb.T)

        # compute U
        U = torch.sparse.sum(sbbs * self.Hint)

        # compute n
        RDensity = torch.zeros(self.naux*self.norb, 2, self.naux*self.norb, 2, device=self.device)
        for a in range(self.naux * self.norb):
            for s in range(2):
                for b in range(a,self.naux * self.norb):
                    if a == b:
                        start = s
                    else:
                        start = 0

                    for s_ in range(start,2):
                        v = torch.sparse.sum(
                            sb * torch.sparse.mm(
                                torch.sparse.mm(sb, annis[(self.naux*self.norb, a, s)].T), annis[(self.naux*self.norb, b, s_)]
                                )
                            )
                        
                        RDensity[a, s, b, s_] = v
                        RDensity[b, s_, a, s] = v

        # compute R
        B = torch.zeros(self.norb, 2, self.naux*self.norb, 2, device=self.device)
        for alpha in range(self.norb):
            for s in range(2):
                for a in range(self.naux*self.norb):
                    for s_ in range(2):
                        B[alpha, s, a, s_] = torch.sparse.sum(
                            sb * torch.sparse.mm(
                                annis[(self.norb, alpha, s)].T, torch.sparse.mm(sb, annis[(self.naux*self.norb, a, s_)])
                                )
                            )

        # alpha, a = T(R(c,alpha)) * sqrt[(RD(1-RD))(ca)]
        A = RDensity.reshape(self.aux_spinorb, self.aux_spinorb)
        A = A @ (torch.eye(self.aux_spinorb, device=self.device)-A)
        A = symsqrt(A).T
        B = B.reshape(self.phy_spinorb, self.aux_spinorb).T
        R = torch.linalg.solve(A=A, B=B).reshape(self.naux*self.norb, 2, self.norb, 2)

        return U, RDensity, R

    def check_Ntotal(self):
        sb = self.SBamp / torch.norm(self.SBamp)
        # vnorm = self.SBamp.norm()
        # sb = torch.cat([2*self.SBamp, vnorm.unsqueeze(0)**2-1]) / (vnorm**2+1)
        sb = torch.sparse_coo_tensor(self.indices, sb, self.size)

        

        sbbs = torch.sparse.mm(sb, sb.T)

        Ntotal = 0
        for a in range(self.naux * self.norb):
            for s in range(2):
                Ntotal += torch.sparse.sum(
                        sb * torch.sparse.mm(
                    torch.sparse.mm(sb, annis[(self.naux*self.norb, a, s)].T), annis[(self.naux*self.norb, a, s)]
                    )
                )
        
        for alpha in range(self.norb):
            for s in range(2):
                Ntotal -= torch.sparse.sum(sbbs * torch.sparse.mm(annis[(self.norb, alpha, s)].T, annis[(self.norb, alpha, s)]))

        return Ntotal
    
    def check_sbbs(self):
        sb = self.SBamp / torch.norm(self.SBamp)
        # vnorm = self.SBamp.norm()
        # sb = torch.cat([2*self.SBamp, vnorm.unsqueeze(0)**2-1]) / (vnorm**2+1)
        sb = torch.sparse_coo_tensor(self.indices, sb, self.size).coalesce()

        sbbs = torch.sparse.mm(sb, sb.T)

        return sbbs


    def check_B(self):
        sb = self.SBamp / torch.norm(self.SBamp)
        # vnorm = self.SBamp.norm()
        # sb = torch.cat([2*self.SBamp, vnorm.unsqueeze(0)**2-1]) / (vnorm**2+1)
        sb = torch.sparse_coo_tensor(self.indices, sb, self.size)

        B = torch.zeros(self.norb, 2, self.naux*self.norb, 2, device=self.device)
        for alpha in range(self.norb):
            for s in range(2):
                for a in range(self.naux*self.norb):
                    for s_ in range(2):
                        B[alpha, s, a, s_] = torch.sparse.sum(
                            sb * torch.sparse.mm(
                                annis[(self.norb, alpha, s)].T, torch.sparse.mm(sb, annis[(self.naux*self.norb, a, s_)])
                                )
                            )
        
        B = B.reshape(self.phy_spinorb, self.aux_spinorb)

        return B

    def check_A(self):
        sb = self.SBamp / torch.norm(self.SBamp)
        sb = torch.sparse_coo_tensor(self.indices, sb, self.size)

        RDensity = torch.zeros(self.naux*self.norb, 2, self.naux*self.norb, 2, device=self.device)
        for a in range(self.naux * self.norb):
            for s in range(2):
                for b in range(a,self.naux * self.norb):
                    if a == b:
                        start = s
                    else:
                        start = 0

                    for s_ in range(start,2):
                        v = torch.sparse.sum(
                            sb * torch.sparse.mm(
                                torch.sparse.mm(sb, annis[(self.naux*self.norb, a, s)].T), annis[(self.naux*self.norb, b, s_)]
                                )
                            )
                        
                        RDensity[a, s, b, s_] = v
                        RDensity[b, s_, a, s] = v

        A = RDensity.reshape(self.aux_spinorb, self.aux_spinorb)
        A = A @ (torch.eye(self.aux_spinorb, device=self.device)-A)
        A = symsqrt(A).T

        return A

    def compute_density(self):
        sb = self.SBamp / torch.norm(self.SBamp)
        # vnorm = self.SBamp.norm()
        # sb = torch.cat([2*self.SBamp, vnorm.unsqueeze(0)**2-1]) / (vnorm**2+1)
        sb = torch.sparse_coo_tensor(self.indices, sb, self.size).coalesce()

        # compute n
        RDensity = torch.zeros(self.norb, 2, self.norb, 2, device=self.device)
        for alpha in range(self.norb):
            for s in range(2):
                for beta in range(alpha,self.norb):
                    if alpha == beta:
                        start = s
                    else:
                        start = 0
                    for s_ in range(start,2):
                        v = torch.sparse.sum(
                            sb * torch.sparse.mm(
                                annis[(self.norb, alpha, s)].T, torch.sparse.mm(annis[(self.norb, beta, s_)], sb)
                                )
                            )
                        
                        RDensity[alpha, s, beta, s_] = v
                        RDensity[beta, s_, alpha, s] = v

        return RDensity


class gGAMultiOrb(nn.Module):
    def __init__(
            self, 
            norbs, 
            naux:int=1, 
            Hint_params:list=[],
            device: Union[str, torch.device] = torch.device("cpu")
            ):
        super(gGAMultiOrb, self).__init__()
        self.norbs = norbs
        self.naux = naux
        self.device = device
        self.dtype = torch.get_default_dtype()
        self.orbcount = sum(norbs)
        self.singleOrbs = nn.ModuleList([gGASingleOrb(norb, naux, Hint_params[i], device=device) for i, norb in enumerate(norbs)])

        for k,v in annis.items():
            annis[k] = v.to(self.device)

    def forward(self):
        U = 0
        RDensity = []
        R = []
        for singleOrb in self.singleOrbs:
            U_, RDensity_, R_ = singleOrb()
            U += U_
            RDensity.append(RDensity_)
            R.append(R_)
        
        RD = []
        RS = []
        for i in range(2):
            for j in range(2):
                RD.append(torch.block_diag(*[RDensity[k][:, i, :, j] for k in range(len(self.norbs))]))
                RS.append(torch.block_diag(*[R[k][:, i, :, j] for k in range(len(self.norbs))]))
        RD = torch.stack(RD).reshape(2,2,self.orbcount*self.naux,self.orbcount*self.naux).permute(2,0,3,1).contiguous()
        RS = torch.stack(RS).reshape(2,2,self.orbcount*self.naux,self.orbcount).permute(2,0,3,1).contiguous()

        return U, RD, RS

    def compute_density(self):
        density = []
        for singleOrb in self.singleOrbs:
            dd = singleOrb.compute_density()
            density.append(dd)

        den = []
        for i in range(2):
            for j in range(2):
                den.append(torch.block_diag(*[density[k][:, i, :, j] for k in range(len(self.norbs))]))
        
        den = torch.stack(den).reshape(2,2,self.orbcount,self.orbcount).permute(2,0,3,1).contiguous()
        
        return den

def symsqrt(matrix): # this may returns nan grade when the eigenvalue of the matrix is very degenerated.
    """Compute the square root of a positive definite matrix."""
    _, s, v = safeSVD(matrix)
    # _, s, v = torch.svd(matrix)
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)