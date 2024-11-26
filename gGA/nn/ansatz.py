import torch
from functorch import grad
from typing import Tuple, Union, Dict
from gGA.model.interaction import slater_kanamori
from gGA.operator import Slater_Kanamori, create_d, annihilate_d, create_u, annihilate_u
from gGA.model.operators import generate_basis, generate_basis_minimized, states_to_indices, generate_product_basis
from gGA.model.operators import annis
from gGA.utils.safe_svd import safeSVD
from gGA.data import OrbitalMapper
from gGA.utils.constants import atomic_num_dict_r
import scipy as sp
import numpy as np
from gGA.utils.tools import real_hermitian_basis

# single means orbital belong to single angular momentum or subspace. for example, s, p, d, d_t2g, etc.
class gGASingleOrb(object):
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
        self.Hint_params = Hint_params
        
        assert naux >= 1

        # here we decompose the basis set, since the constraint are that the physical+auxiliary basis set is half filled.
        # we have basis for the physical system with occupation number [0, 1, ..., 2*norb]
        # and have basis for the auxiliary system with occupation number [(naux+1)*norb, (naux+1)*norb-1, ..., (naux-1)*norb]

        # set up the prompt independent variables: RDM of the auxiliary system, R matrix
        self.hermit_basis = real_hermitian_basis(self.aux_spinorb)

        self.RDM_aux = torch.randn(int(self.aux_spinorb*(self.aux_spinorb+1)/2), dtype=self.dtype, device=self.device)

        self.R = torch.randn(self.aux_spinorb, self.phy_spinorb, dtype=self.dtype, device=self.device)
        # Since we know R @ R.T is a diagonal matrix, can we use this properties to accelerate convergence?

        # setup langrangian multipliers
        self.lag_den_emb = torch.randn(int(self.aux_spinorb*(self.aux_spinorb+1)/2), dtype=self.dtype, device=self.device)
        self.lag_R = torch.randn(self.aux_spinorb, self.phy_spinorb, dtype=self.dtype, device=self.device)

        self._Hemb_uptodate = False
        self._decouple_bath = False
        self._nature_orbital = False

        def LAM_C_fn(RDM_aux, D, R, hermit_basis):
            RDM = (hermit_basis * RDM_aux[:,None,None]).sum(dim=0)
            out = RDM @ (torch.eye(RDM.shape[0])-RDM)
            out = (symsqrt(out) @ D @ R.T).sum()

            return out

        self.lag_update_fn = grad(LAM_C_fn, argnum=0)

    def update(self, LAM_so, solver="ED", decouple_bath: bool=False, natural_orbital: bool=False):
        # update lag_dem_emb, LAM_so is single-orbital Lambda lagrangian multipliers in matrix form
        lag_den_qp = (self.hermit_basis * LAM_so[None,:,:]).sum(dim=(1,2))
        self.lag_den_emb = - self.lag_update_fn(self.RDM_aux, self.D, self.R, self.hermit_basis) - lag_den_qp
        self._Hemb_uptodate = False
        
        # update R, RDM
        R, RDM = self.solve_Hemb(solver, decouple_bath, natural_orbital)
        self.R = R
        self.RDM_aux = (RDM[None,:,:] * self.hermit_basis).sum(dim=(1,2))

        self._Hemb_uptodate = False

        return R, RDM

    @property
    def RDM(self):
        return (self.hermit_basis * self.RDM_aux[:,None,None]).sum(dim=0)
    
    @property
    def LAM_C(self):
        return (self.hermit_basis * self.lag_den_emb[:,None,None]).sum(dim=0)

    @property
    def D(self):
        return self.lag_R

    def get_Hemb(self, decouple_bath: bool=False, natural_orbital: bool=False):
        if decouple_bath != self._decouple_bath or natural_orbital != self._nature_orbital or not self._Hemb_uptodate:
            return self._construct_Hemb(decouple_bath, natural_orbital)

        else:
            return self._Hemb


    def _construct_Hemb(self, decouple_bath: bool=False, natural_orbital: bool=False):
        # construct the embedding Hamiltonian
        assert decouple_bath + (not natural_orbital), "Not support natural orbital representation of bath when bath is not decoupled."
        self.decouple_bath = decouple_bath
        self.natrual_orbital = natural_orbital

        # constructing the kinetical part T
        T = torch.zeros(self.aux_spinorb+self.phy_spinorb, self.aux_spinorb+self.phy_spinorb, device=self.device)
        t = self.Hint_params.get("t", 1.0)
        if isinstance(t, torch.Tensor):
            if t.shape[0] == self.norb: # spin degenerate
                T[:self.norb*2, self.norb*2:] = torch.block_diag([t,t])
            else:
                assert t.shape[0] == self.norb*2
                T[:self.norb*2, :self.norb*2] = t
        else:
            assert isinstance(t, (int, float))
            T[:self.norb*2, self.norb*2:] = torch.eye(self.norb*2, device=self.device) * t
        
        T[:self.phy_spinorb, self.phy_spinorb:] = self.D.T
        T[self.phy_spinorb:, :self.phy_spinorb] = self.D
        T[self.phy_spinorb:, self.phy_spinorb:] = self.LAM_C

        Hint_params = self.Hint_params.copy()
        Hint_params["t"] = T.detach().cpu().numpy()

        if not decouple_bath:
            self._Hemb = Slater_Kanamori(
                nsites=self.norb*(self.naux+1),
                n_noninteracting=self.norb*self.naux,
                **Hint_params
            ) 
            # we should notice that the spinorbital are not adjacent in the quspin hamiltonian, 
            # so properties computed from this need to be transformed.
        else:
            if natural_orbital:
                pass
            # TODO, do the transformation latter
            raise NotImplementedError

        self._Hemb_uptodate = True
        return self._Hemb


    def solve_Hemb(self, solver="ED", decouple_bath: bool=False, natural_orbital: bool=False):
        # handeling the natural_orbital and decoupled transformation here
        if solver == "ED":
            R, RDM = self._solve_Hemb_ED(decouple_bath, natural_orbital)

        else:
            raise NotImplementedError

        return R, RDM

    def _solve_Hemb_ED(self, decouple_bath: bool=False, natural_orbital: bool=False):
        Hemb = self.get_Hemb(decouple_bath, natural_orbital)
        nsites = self.norb*(self.naux+1)
        Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]
        val, vec = Hemb.diagonalize(
            nsites=nsites, 
            Nparticle=Nparticle
        )

        # compute RDM
        RDM = np.zeros((self.naux*self.norb, 2, self.naux*self.norb, 2))
        for a in range(self.norb, (self.naux+1) * self.norb):
            for s in range(2):
                for b in range(a, (self.naux+1) * self.norb):
                    if a == b:
                        start = s
                    else:
                        start = 0

                    for s_ in range(start,2):
                        if s == 0 and s_ == 0:
                            op = create_u(self.norb*(self.naux+1), a) * annihilate_u(self.norb*(self.naux+1), b)
                        elif s == 1 and s_ == 1:
                            op = create_d(self.norb*(self.naux+1), a) * annihilate_d(self.norb*(self.naux+1), b)
                        elif s == 0 and s_ == 1:
                            op = create_u(self.norb*(self.naux+1), a) * annihilate_d(self.norb*(self.naux+1), b)
                        else:
                            op = create_d(self.norb*(self.naux+1), a) * annihilate_u(self.norb*(self.naux+1), b)
                        
                        v = op.get_quspin_op(nsites, Nparticle).expt_value(vec)
                        RDM[a-self.norb, s, b-self.norb, s_] = v
                        RDM[b-self.norb, s_, a-self.norb, s] = v

        RDM = RDM.reshape(self.aux_spinorb, self.aux_spinorb)

        # compute B
        # compute R
        B = np.zeros((self.norb, 2, self.naux*self.norb, 2))
        for alpha in range(self.norb):
            for s in range(2):
                for a in range(self.norb, (self.naux+1) * self.norb):
                    for s_ in range(2):
                        if s == 0 and s_ == 0:
                            op = create_u(self.norb*(self.naux+1), alpha) * annihilate_u(self.norb*(self.naux+1), a)
                        elif s == 1 and s_ == 1:
                            op = create_d(self.norb*(self.naux+1), alpha) * annihilate_d(self.norb*(self.naux+1), a)
                        elif s == 0 and s_ == 1:
                            op = create_u(self.norb*(self.naux+1), alpha) * annihilate_d(self.norb*(self.naux+1), a)
                        else:
                            op = create_d(self.norb*(self.naux+1), alpha) * annihilate_u(self.norb*(self.naux+1), a)

                        v = op.get_quspin_op(nsites, Nparticle).expt_value(vec)
                        B[alpha, s, a-self.norb, s_] = v

        # alpha, a = T(R(c,alpha)) * sqrt[(RD(1-RD))(ca)]
        A = RDM @ (np.eye(self.aux_spinorb)-RDM)
        A = A + np.eye(self.aux_spinorb) * 1e-6 # to avoid negative sqrt
        A = sp.linalg.sqrtm(A).T
        B = B.reshape(self.phy_spinorb, self.aux_spinorb).T
        R = sp.linalg.solve(a=A, b=B).reshape(self.naux*self.norb*2, self.norb*2)
        
        return torch.from_numpy(R, device=self.device), torch.from_numpy(RDM,  device=self.device)

    def _solve_Hemb_VMC(self):
        pass

    def _solve_Hemb_DMRG(self):
        pass

    def _solve_Hemb_TTN(self):
        pass

    def check(self):
        pass

    def RDM_phy(self):
        pass


class gGAMultiOrb(object):
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
        self.singleOrbs = [gGASingleOrb(norb, naux, Hint_params[i], device=device) for i, norb in enumerate(norbs)]

    def update(self, LAM_mo, solver="ED", decouple_bath: bool=False, natural_orbital: bool=False):
        RDMs = []
        Rs = []
        co = 0
        # LAM_mo, Lambda lagrangian multipliers of multi-orbital in matrix form, should have shape (sum(norbs)*naux*2, sum(norbs)*naux*2)
        for singleOrb in self.singleOrbs:
            LAM_so = LAM_mo[co:co+singleOrb.aux_spinorb, co:co+singleOrb.aux_spinorb] # [aux_spinorb, aux_spinorb]
            co += singleOrb.aux_spinorb
            R, RDM = singleOrb.update(LAM_so, solver, decouple_bath, natural_orbital)
            RDMs.append(RDM)
            Rs.append(R)

        return torch.block_diag(*Rs), torch.block_diag(*RDMs)

    @property
    def RDM(self):
        return torch.block_diag(*[singleOrb.RDM for singleOrb in self.singleOrbs])
    
    @property
    def LAM_C(self):
        return torch.block_diag(*[singleOrb.LAM_C for singleOrb in self.singleOrbs])

    @property
    def D(self):
        return torch.block_diag(*[singleOrb.D for singleOrb in self.singleOrbs])

    @property
    def R(self):
        return torch.block_diag(*[singleOrb.R for singleOrb in self.singleOrbs])

class gGAtomic(object):
    def __init__(self, basis, atomic_number,idx_intorb, Hint_params, naux, device):
        self.basis = basis
        self.atomic_number = atomic_number
        self.naux = naux
        self.idp_phy = OrbitalMapper(basis=basis, device=device, spin_deg=False)
        self.idx_intorb = idx_intorb
        self.device = device

        # do some orbital statistics
        nauxorbs = self.idp_phy.listnorbs.copy() # {'C': [1, 3], 'Si': [1, 1, 3, 3, 5]}
        self.intorbs = {}
        self.nintphy = {}
        self.nintaux = {}
        for sym in self.idx_intorb:
            self.intorbs[sym] = [self.idp_phy.basis[sym][idx] for idx in self.idx_intorb[sym]]
            self.nintphy[sym] = [nauxorbs[sym][idx] for idx in self.idx_intorb[sym]]
            self.nintaux[sym] = [nauxorbs[sym][idx] * naux for idx in self.idx_intorb[sym]]
        
        self.interact_ansatz = []
        for ia in range(len(atomic_number)):
            sym = atomic_num_dict_r[int(atomic_number[ia])]
            intorb_a = [nauxorbs[sym][i] for i in idx_intorb[int(atomic_number[ia])]]
            self.interact_ansatz.append(    
                gGAMultiOrb(
                    norbs=intorb_a,
                    naux=naux, 
                    Hint_params=Hint_params[int(ia)],
                    device=device
                    )
                ) # TODO: The most computational demanding part, would be perfect if it is parallizable

        # generate the idp for gost system
        for sym in nauxorbs:
            for i in self.idx_intorb[sym]:
                nauxorbs[sym][i] *= naux # TODO: do we need to consider the order of orbitals? 
                # Seems the increase of some number of orbitals would also change the order in idp
        self.nauxorbs = nauxorbs

        self.idp_intaux = OrbitalMapper(basis=self.nintaux, device=device, spin_deg=False) # since in gGA, we only consider spin system
        self.idp_intphy = OrbitalMapper(basis=self.intorbs, device=device, spin_deg=False)

    def update(self, LAM, solver="ED", decouple_bath: bool=False, natural_orbital: bool=False):
        # LAM should have a stacked matrix format, with shape [natoms, sum(nintorbs)*naux*2, sum(nintorbs)*naux*2]
        RDM = torch.zeros((len(self.atomic_number), self.idp_intaux.norb*2, self.idp_intaux.norb*2), device=self.device)
        R = torch.zeros((len(self.atomic_number), self.idp_intphy.norb*2, self.idp_intphy.norb*2), device=self.device)
        for ia, an in enumerate(self.atomic_number):
            sym = atomic_num_dict_r[int(an)]
            at = self.idp_phy.chemical_symbol_to_type[sym]
            mask_phy = self.idp_intphy.mask_to_basis[at]
            mask_phy = torch.arange(len(mask_phy), device=self.device)[mask_phy]
            mask_aux = self.idp_intaux.mask_to_basis[at]
            mask_aux = torch.arange(len(mask_aux), device=self.device)[mask_aux]
            LAM_a = LAM[ia, mask_aux.unsqueeze(1), mask_aux.unsqueeze(0)]
            Rs, RDMs = self.interact_ansatz[ia].update(LAM_a, solver, decouple_bath, natural_orbital)
            # update the R, RDM for each atomic system
            RDM[ia, mask_aux.unsqueeze(1), mask_aux.unsqueeze(0)] += RDMs
            R[ia, mask_aux.unsqueeze(1), mask_phy.unsqueeze(0)] += Rs
        
        RDM.contiguous()
        R.contiguous()
        
        return R, RDM
    
    def update_D_from_kin(self):
        pass

    @property
    def R(self):
        r = torch.zeros((len(self.atomic_number), self.idp_intphy.norb*2, self.idp_intphy.norb*2), device=self.device)
        for ia, an in enumerate(self.atomic_number):
            sym = atomic_num_dict_r[int(an)]
            at = self.idp_phy.chemical_symbol_to_type[sym]
            mask_phy = self.idp_intphy.mask_to_basis[at]
            mask_phy = torch.arange(len(mask_phy), device=self.device)[mask_phy]
            mask_aux = self.idp_intaux.mask_to_basis[at]
            mask_aux = torch.arange(len(mask_aux), device=self.device)[mask_aux]

            r[ia, mask_aux.unsqueeze(1), mask_phy.unsqueeze(0)] += self.interact_ansatz[ia].R
        
        r.contiguous()

        return r

    @property
    def RDM(self):
        rdm = torch.zeros((len(self.atomic_number), self.idp_intaux.norb*2, self.idp_intaux.norb*2), device=self.device)
        for ia, an in enumerate(self.atomic_number):
            sym = atomic_num_dict_r[int(an)]
            at = self.idp_phy.chemical_symbol_to_type[sym]
            mask_aux = self.idp_intaux.mask_to_basis[at]
            mask_aux = torch.arange(len(mask_aux), device=self.device)[mask_aux]

            rdm[ia, mask_aux.unsqueeze(1), mask_aux.unsqueeze(0)] += self.interact_ansatz[ia].RDM
        
        rdm.contiguous()

        return rdm

    @property
    def D(self):
        d = torch.zeros((len(self.atomic_number), self.idp_intphy.norb*2, self.idp_intphy.norb*2), device=self.device)
        for ia, an in enumerate(self.atomic_number):
            sym = atomic_num_dict_r[int(an)]
            at = self.idp_phy.chemical_symbol_to_type[sym]
            mask_phy = self.idp_intphy.mask_to_basis[at]
            mask_phy = torch.arange(len(mask_phy), device=self.device)[mask_phy]
            mask_aux = self.idp_intaux.mask_to_basis[at]
            mask_aux = torch.arange(len(mask_aux), device=self.device)[mask_aux]

            d[ia, mask_aux.unsqueeze(1), mask_phy.unsqueeze(0)] += self.interact_ansatz[ia].D
        
        d.contiguous()

        return d

    @property
    def LAM_C(self):
        lam_c = torch.zeros((len(self.atomic_number), self.idp_intaux.norb*2, self.idp_intaux.norb*2), device=self.device)
        for ia, an in enumerate(self.atomic_number):
            sym = atomic_num_dict_r[int(an)]
            at = self.idp_phy.chemical_symbol_to_type[sym]
            mask_aux = self.idp_intaux.mask_to_basis[at]
            mask_aux = torch.arange(len(mask_aux), device=self.device)[mask_aux]

            lam_c[ia, mask_aux.unsqueeze(1), mask_aux.unsqueeze(0)] += self.interact_ansatz[ia].LAM_C
        
        lam_c.contiguous()

        return lam_c


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