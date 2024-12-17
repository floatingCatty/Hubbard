import torch
from typing import Tuple, Union, Dict
from gGA.operator import Slater_Kanamori, create_d, annihilate_d, create_u, annihilate_u
from gGA.utils.safe_svd import safeSVD
from gGA.data import OrbitalMapper
from gGA.utils.constants import atomic_num_dict_r, atomic_num_dict
import scipy as sp
import numpy as np
import copy
from gGA.utils.tools import real_hermitian_basis_nspin, real_trans_basis_nspin

# single means orbital belong to single angular momentum or subspace. for example, s, p, d, d_t2g, etc.
class gGASingleOrb(object):
    def __init__(
            self, 
            norb, 
            nocc,
            naux:int=1, 
            intparam:dict={},
            nspin: int=1, # 1 for spin degenerate, 2 for collinear spin polarized, 4 for non-collinear spin polarization
            device: Union[str, torch.device] = torch.device("cpu")
            ):
        
        super(gGASingleOrb, self).__init__()
        self.dtype = torch.get_default_dtype()
        self.device = device
        self.norb = norb
        self.naux = naux
        self.nocc = nocc
        self.nauxorb = naux * norb
        self.nspin = nspin

        self.phy_spinorb = self.norb*2
        self.aux_spinorb = self.naux*self.norb*2

        self.intparam = intparam
        self._t = 0
        
        assert naux >= 1

        # here we decompose the basis set, since the constraint are that the physical+auxiliary basis set is half filled.
        # we have basis for the physical system with occupation number [0, 1, ..., 2*norb]
        # and have basis for the auxiliary system with occupation number [(naux+1)*norb, (naux+1)*norb-1, ..., (naux-1)*norb]

        # set up the prompt independent variables: RDM of the auxiliary system, R matrix
        
        """
            Parameterization doc:
                nspin = 1: spin degenerate [spin_up/down]
                nspin = 2: spin collinear [spin_up, spin_down]
                nspin = 4: spin non-collinear [spin_up, spin_down, spin_updown, spin_downup]
            The real parameter shoud be infered as property: 
                R, RDM, LAM, LAM_C
            The process is written as a basis production.
        """


        ################## initialize basis for different spin setting ##############################
        self.hermit_basis = real_hermitian_basis_nspin(self.aux_spinorb, self.nspin)
        self.trans_basis = real_trans_basis_nspin(self.aux_spinorb, self.phy_spinorb, self.nspin)
        
        self._rdm = torch.zeros(int(self.nspin*self.nauxorb*(self.norb+1)/2), dtype=self.dtype, device=self.device)
        R = torch.rand(self.aux_spinorb, self.phy_spinorb, device=self.device)
        self.update_R(R=R)
        # self._r = torch.ones(self.nspin*self.nauxorb*self.norb, dtype=self.dtype, device=self.device) * 0.5
        # Since we know R @ R.T is a diagonal matrix, can we use this properties to accelerate convergence?

        # setup langrangian multipliers
        LAM = torch.zeros((self.naux * self.norb * 2,), device=self.device)
        LAM[:(naux//2)*self.norb*2] = 0.1
        LAM[-(naux//2)*self.norb*2:] = -0.1
        LAM = torch.diag(LAM)
        self.update_LAM(LAM=LAM)
        self._lamc = torch.zeros(int(self.nspin*self.nauxorb*(self.nauxorb+1)/2), dtype=self.dtype, device=self.device)
        self._d = torch.zeros(self.nspin*self.nauxorb*self.norb, dtype=self.dtype, device=self.device)
        self._Hemb_uptodate = False
        self._decouple_bath = False
        self._nature_orbital = False

    def update(self, t, E_fermi, solver="ED", decouple_bath: bool=False, natural_orbital: bool=False):
        self._lamc = calc_lam_c(self.R, self._lam, self.RDM, self.D, self.hermit_basis)

        self._Hemb_uptodate = False
        
        # update R, RDM
        R, RDM = self.solve_Hemb(t, E_fermi, solver, decouple_bath, natural_orbital)
        self.update_RDM(RDM=RDM)
        self.update_R(R=R)
        self._lam = calc_lam_c(self.R, self._lamc, self.RDM, self.D, self.hermit_basis)
        
        return self.R, self.LAM # R, LAM
    
    def fix_gauge(self):
        # fix gauge of lambda and R
        R, LAM = self.R, self.LAM
        eigval, eigvec = torch.linalg.eigh(LAM)
        LAM = torch.diag(eigval)
        R = eigvec.T @ R

        mat = torch.ones_like(R[:,0])
        mat[R[:,0].lt(0)] = -1
        mat = torch.diag(mat)
        R = mat @ R
        LAM = mat @ LAM @ mat.T

        idx = torch.argsort(R[:,0], descending=True)
        mat = torch.eye(R.shape[0])[idx]
        LAM = mat @ LAM @ mat.T
        R = mat @ R

        self.update_LAM(LAM=LAM)
        self.update_R(R=R)

        self._Hemb_uptodate = False

        return True


    @property
    def RDM(self):
        return (self.hermit_basis * self._rdm[:,None,None]).sum(dim=0)
    
    def update_RDM(self, RDM):
        self._rdm = (RDM.unsqueeze(0) * self.hermit_basis).sum(dim=(1,2))
        self._Hemb_uptodate = False
        return True
    
    @property
    def R(self):
        return (self.trans_basis * self._r[:, None, None]).sum(0)

    def update_R(self, R):
        self._r = (R.unsqueeze(0) * self.trans_basis).sum(dim=(1,2))
        self._Hemb_uptodate = False
        return True

    @property
    def D(self):
        return (self.trans_basis * self._d[:, None, None]).sum(0)

    def update_D(self, D):
        self._d = (D.unsqueeze(0) * self.trans_basis).sum(dim=(1,2))
        self._Hemb_uptodate = False
        return True

    @property
    def LAM_C(self):
        return (self.hermit_basis * self._lamc[:,None,None]).sum(dim=0)
    
    def update_LAM_C(self, LAM_C):
        self._lamc = (self.hermit_basis * LAM_C.unsqueeze(0)).sum(dim=(1,2))
        self._Hemb_uptodate = False
        return True

    @property
    def LAM(self):
        return (self.hermit_basis * self._lam[:,None,None]).sum(dim=0)
    
    def update_LAM(self, LAM):
        self._lam = (self.hermit_basis * LAM.unsqueeze(0)).sum(dim=(1,2))
        self._Hemb_uptodate = False
        return True


    def get_Hemb(self, t, E_fermi, decouple_bath: bool=False, natural_orbital: bool=False):
        if decouple_bath != self._decouple_bath or natural_orbital != self._nature_orbital or not self._Hemb_uptodate or abs(self._t - t).max() > 1e-6:
            return self._construct_Hemb(t, E_fermi, decouple_bath, natural_orbital)

        else:
            return self._Hemb


    def _construct_Hemb(self, t, E_fermi, decouple_bath: bool=False, natural_orbital: bool=False):
        # construct the embedding Hamiltonian
        assert decouple_bath + (not natural_orbital), "Not support natural orbital representation of bath when bath is not decoupled."
        self.decouple_bath = decouple_bath
        self.natrual_orbital = natural_orbital

        # constructing the kinetical part T
        T = torch.zeros(self.aux_spinorb+self.phy_spinorb, self.aux_spinorb+self.phy_spinorb, device=self.device)
        if isinstance(t, torch.Tensor):
            if t.shape[0] == self.norb: # spin degenerate
                assert self.nspin == 1
                T[:self.norb*2, :self.norb*2] = torch.block_diag(t,t).reshape(2,self.norb,2,-1).permute(1,0,3,2).reshape(self.norb*2,-1)
            else:
                assert self.nspin > 1
                assert t.shape[0] == self.norb*2, "the hopping t's shape does not match either spin denegrate case or spin case"
                T[:self.norb*2, :self.norb*2] = t
        else:
            assert isinstance(t, (int, float))
            T[:self.norb*2, self.norb*2:] = torch.eye(self.norb*2, device=self.device) * t
        
        T[:self.norb*2, :self.norb*2] -= E_fermi * torch.eye(self.norb*2, device=self.device)
        T[:self.phy_spinorb, self.phy_spinorb:] = self.D.T
        T[self.phy_spinorb:, :self.phy_spinorb] = self.D
        T[self.phy_spinorb:, self.phy_spinorb:] = - self.LAM_C
        self._t = T.clone()

        intparam = copy.deepcopy(self.intparam)
        intparam["t"] = T.detach().cpu().numpy()

        if not decouple_bath:
            self._Hemb = Slater_Kanamori(
                nsites=self.norb*(self.naux+1),
                n_noninteracting=self.norb*self.naux,
                **intparam
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


    def solve_Hemb(self, t, E_fermi, solver="ED", decouple_bath: bool=False, natural_orbital: bool=False):
        # handeling the natural_orbital and decoupled transformation here
        if solver == "ED":
            R, RDM = self._solve_Hemb_ED(t, E_fermi, decouple_bath, natural_orbital)
        else:
            raise NotImplementedError
        
        return R, RDM

    def _solve_Hemb_ED(self, t, E_fermi, decouple_bath: bool=False, natural_orbital: bool=False):
        Hemb = self.get_Hemb(t, E_fermi, decouple_bath, natural_orbital)
        nsites = self.norb*(self.naux+1)
        # Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]
        if self.nspin == 1:
            # Nparticle = [(self.norb*(self.naux+1)-i,i) for i in range(0,self.norb*(self.naux+1)+1)]
            Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]
        else:
            Nparticle = [(self.norb*(self.naux+1)-i,i) for i in range(0,self.norb*(self.naux+1)+1)]
        val, vec = Hemb.diagonalize(
            nsites=nsites, 
            Nparticle=Nparticle
        )

        # # compute RDM
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
                            op = create_u(nsites, a) * annihilate_u(nsites, b)
                        elif s == 1 and s_ == 1:
                            op = create_d(nsites, a) * annihilate_d(nsites, b)
                        elif s == 0 and s_ == 1:
                            op = create_u(nsites, a) * annihilate_d(nsites, b)
                        else:
                            op = create_d(nsites, a) * annihilate_u(nsites, b)
                        
                        v = op.get_quspin_op(nsites, Nparticle).expt_value(vec)
                        RDM[a-self.norb, s, b-self.norb, s_] = v
                        RDM[b-self.norb, s_, a-self.norb, s] = v

        RDM = np.eye(self.aux_spinorb) - RDM.reshape(self.aux_spinorb, self.aux_spinorb)

        # compute B
        # compute R
        B = np.zeros((self.norb, 2, self.naux*self.norb, 2))
        for alpha in range(self.norb):
            for s in range(2):
                for a in range(self.norb, (self.naux+1) * self.norb):
                    for s_ in range(2):
                        if s == 0 and s_ == 0:
                            op = create_u(nsites, alpha) * annihilate_u(nsites, a)
                        elif s == 1 and s_ == 1:
                            op = create_d(nsites, alpha) * annihilate_d(nsites, a)
                        elif s == 0 and s_ == 1:
                            op = create_u(nsites, alpha) * annihilate_d(nsites, a)
                        else:
                            op = create_d(nsites, alpha) * annihilate_u(nsites, a)

                        v = op.get_quspin_op(nsites, Nparticle).expt_value(vec)
                        B[alpha, s, a-self.norb, s_] = v

        # alpha, a = T(R(c,alpha)) * sqrt[(RD(1-RD))(ca)]
        A = RDM @ (np.eye(self.aux_spinorb)-RDM)
        A = sp.linalg.sqrtm(A).T
        B = B.reshape(self.phy_spinorb, self.aux_spinorb).T
        R = sp.linalg.solve(a=A, b=B).reshape(self.aux_spinorb, self.phy_spinorb)
        
        R = torch.from_numpy(R).to(device=self.device, dtype=self.dtype)
        RDM = torch.from_numpy(RDM).to(device=self.device, dtype=self.dtype)

        return R, RDM

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
            noccs,
            naux:int=1,
            nspin:int=1,
            intparams:list=[],
            device: Union[str, torch.device] = torch.device("cpu")
            ):
        super(gGAMultiOrb, self).__init__()
        self.norbs = norbs
        self.noccs = noccs
        self.naux = naux
        self.device = device
        self.nspin = nspin
        self.dtype = torch.get_default_dtype()
        self.orbcount = sum(norbs)
        self.singleOrbs = [gGASingleOrb(norb, noccs[i], naux, intparams[i], nspin=nspin, device=device) for i, norb in enumerate(norbs)]

    def update(self, t, E_fermi, solver="ED", decouple_bath: bool=False, natural_orbital: bool=False):
        LAMs = []
        Rs = []
        co = 0
        # LAM_mo, Lambda lagrangian multipliers of multi-orbital in matrix form, should have shape (sum(norbs)*naux*2, sum(norbs)*naux*2)
        for iso, singleOrb in enumerate(self.singleOrbs):
            t_so = t[iso] # [aux_spinorb, aux_spinorb]
            co += singleOrb.aux_spinorb
            R, LAM = singleOrb.update(t_so, E_fermi, solver, decouple_bath, natural_orbital)
            LAMs.append(LAM)
            Rs.append(R)

        return Rs, LAMs
    
    def fix_gauge(self):
        for i in range(len(self.norbs)):
            self.singleOrbs[i].fix_gauge()

        return True

    @property
    def RDM(self):
        return [singleOrb.RDM for singleOrb in self.singleOrbs]

    def update_RDM(self, RDM):
        for i, singleOrb in enumerate(self.singleOrbs):
            singleOrb.update_RDM(RDM[i])

        return True
    
    @property
    def LAM_C(self):
        return [singleOrb.LAM_C for singleOrb in self.singleOrbs]
    
    def update_LAM_C(self, LAM_C):
        for i, singleOrb in enumerate(self.singleOrbs):
            singleOrb.update_LAM_C(LAM_C[i])
        
        return True

    @property
    def LAM(self):
        return [singleOrb.LAM for singleOrb in self.singleOrbs]
    
    def update_LAM(self, LAM):
        for i, singleOrb in enumerate(self.singleOrbs):
            singleOrb.update_LAM(LAM[i])
        
        return True

    @property
    def D(self):
        return [singleOrb.D for singleOrb in self.singleOrbs]

    def update_D(self, D):
        for i, singleOrb in enumerate(self.singleOrbs):
            singleOrb.update_D(D[i])

        return True

    @property
    def R(self):
        return [singleOrb.R for singleOrb in self.singleOrbs]
    
    def update_R(self, R):
        for i, singleOrb in enumerate(self.singleOrbs):
            singleOrb.update_R(R[i])

        return True

class gGAtomic(object):
    def __init__(
            self, 
            basis: dict, 
            atomic_number: torch.Tensor, 
            idx_intorb: Dict[str, list], 
            nocc: Dict[str, list],
            intparams: dict, 
            naux: int, 
            nspin: int,
            device: str,
            spin_deg: bool=True, # it is for the physical system
        ):

        self.basis = basis
        self.atomic_number = atomic_number
        self.naux = naux
        self.nocc = nocc
        self.nspin = nspin
        self.idp_phy = OrbitalMapper(basis=basis, device=device, spin_deg=spin_deg)
        self.idx_intorb = idx_intorb
        self.device = device

        for sym, param in intparams.items():
            assert len(param) == len(idx_intorb[sym]), "Hint_params should have the same length as idx_intorb"

        # do some orbital statistics
        
        self.interact_ansatz = []
        for ia, an in enumerate(atomic_number):
            sym = atomic_num_dict_r[int(atomic_number[ia])]
            intorb_a = [self.idp_phy.listnorbs[sym][i] for i in idx_intorb[atomic_num_dict_r[int(atomic_number[ia])]]]
            intnocc_a = [nocc[sym][i] for i in idx_intorb[atomic_num_dict_r[int(atomic_number[ia])]]]
            self.interact_ansatz.append(    
                gGAMultiOrb(
                    norbs=intorb_a,
                    naux=naux, 
                    nspin=nspin, 
                    intparams=intparams[sym],
                    noccs=intnocc_a,
                    device=device
                    )
                ) # TODO: The most computational demanding part, would be perfect if it is parallizable

        # generate the idp for gost system
        nauxorbs = copy.deepcopy(self.idp_phy.listnorbs) # {'C': [1, 3], 'Si': [1, 1, 3, 3, 5]}
        for sym in nauxorbs:
            for i in self.idx_intorb[sym]:
                nauxorbs[sym][i] = nauxorbs[sym][i] * naux
        self.nauxorbs = nauxorbs

        start_int_orbs = {sym:[] for sym in self.idp_phy.type_names}
        start_phy_orbs = {sym:[] for sym in self.idp_phy.type_names}
        for sym in self.idp_phy.type_names:
            for i in self.idx_intorb[sym]:
                start_int_orbs[sym].append(sum(nauxorbs[sym][:i])*2)
                start_phy_orbs[sym].append(sum(self.idp_phy.listnorbs[sym][:i]))
        self.start_int_orbs = start_int_orbs
        self.start_phy_orbs = start_phy_orbs

    def update(self, t, E_fermi, solver="ED", decouple_bath: bool=False, natural_orbital: bool=False):
        # LAM should have a stacked matrix format, with shape [natoms, sum(nintorbs)*naux*2, sum(nintorbs)*naux*2]
        LAM = {sym:[[torch.zeros((i*2,i*2,), device=self.device) for i in self.idp_phy.listnorbs[sym]]]*int(self.atomic_number.eq(atomic_num_dict[sym]).sum()) 
               for sym in self.idp_phy.type_names}
        R = {sym:[[torch.eye(i*2, device=self.device) for i in self.idp_phy.listnorbs[sym]]]*int(self.atomic_number.eq(atomic_num_dict[sym]).sum()) 
               for sym in self.idp_phy.type_names}
        sfactor = self.idp_phy.spin_factor

        for sym in self.idp_phy.type_names:
            for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
                # ita for the ith atom in sym type, ia is its id in atomic number
                t_a = t[sym][ita]
                assert t_a.shape[1] == self.idp_phy.atom_norb[self.idp_phy.chemical_symbol_to_type[sym]], "Shape of t is not correct!"
                t_a = [t_a[s*sfactor:s*sfactor+self.idp_phy.listnorbs[sym][self.idx_intorb[sym][i]]*sfactor,s*sfactor:s*sfactor+self.idp_phy.listnorbs[sym][self.idx_intorb[sym][i]]*sfactor] for i, s in enumerate(self.start_phy_orbs[sym])]
                Rs, LAMs = self.interact_ansatz[ia].update(t_a, E_fermi, solver, decouple_bath, natural_orbital)
                # update the R, RDM for each atomic system

                for i, into in enumerate(self.idx_intorb[sym]):
                    LAM[sym][ita][into] = LAMs[i]
                    R[sym][ita][into] = Rs[i]
                LAM[sym][ita] = torch.block_diag(*LAM[sym][ita])
                R[sym][ita] = torch.block_diag(*R[sym][ita])
            LAM[sym] = torch.stack(LAM[sym])
            R[sym] = torch.stack(R[sym])
        
            LAM[sym].contiguous()
            R[sym].contiguous()
        
        return R, LAM

    def fix_gauge(self):
        for i in range(len(self.atomic_number)):
            self.interact_ansatz[i].fix_gauge()

        return True

    @property
    def R(self):
        R = {sym:[[torch.eye(i*2, device=self.device) for i in self.idp_phy.listnorbs[sym]]]*int(self.atomic_number.eq(atomic_num_dict[sym]).sum()) 
               for sym in self.idp_phy.type_names}
        for sym in self.idp_phy.type_names:
            for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
                for i, into in enumerate(self.idx_intorb[sym]):
                    R[sym][ita][into] = self.interact_ansatz[ia].R[i]
                R[sym][ita] = torch.block_diag(*R[sym][ita])
            R[sym] = torch.stack(R[sym])

        return R
    
    def update_R(self, R):
        for sym in self.idp_phy.type_names:
            R_split = list(zip(*[
                R[sym][:,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2,self.start_phy_orbs[sym][i]*2:self.start_phy_orbs[sym][i]*2+self.idp_phy.listnorbs[sym][self.idx_intorb[sym][i]]*2] 
                for i, s in enumerate(self.start_int_orbs[sym])]))
            for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
                self.interact_ansatz[ia].update_R(R_split[ita])
        
        return True

    @property
    def RDM(self):
        RDM = {sym:[[torch.eye(i*2, device=self.device) for i in self.idp_phy.listnorbs[sym]]]*int(self.atomic_number.eq(atomic_num_dict[sym]).sum()) 
               for sym in self.idp_phy.type_names}
        for sym in self.idp_phy.type_names:
            for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
                for i, into in enumerate(self.idx_intorb[sym]):
                    RDM[sym][ita][into] = self.interact_ansatz[ia].RDM[i]
                RDM[sym][ita] = torch.block_diag(*RDM[sym][ita])
            RDM[sym] = torch.stack(RDM[sym])

        return RDM

    def update_RDM(self, RDM):
        # RDM should have the same shape as the property RDM
        for sym in self.idp_phy.type_names:
            RDM_split = list(zip(*[
                RDM[sym][:,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2] 
                for i, s in enumerate(self.start_int_orbs[sym])]))

            for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
                self.interact_ansatz[ia].update_RDM(RDM_split[ita])
        
        return True


    @property
    def D(self):
        D = {sym:[[torch.eye(i*2, device=self.device) for i in self.idp_phy.listnorbs[sym]]]*int(self.atomic_number.eq(atomic_num_dict[sym]).sum()) 
               for sym in self.idp_phy.type_names}
        for sym in self.idp_phy.type_names:
            for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
                for i, into in enumerate(self.idx_intorb[sym]):
                    D[sym][ita][into] = self.interact_ansatz[ia].D[i]
                D[sym][ita] = torch.block_diag(*D[sym][ita])
            D[sym] = torch.stack(D[sym])

        return D

    def update_D(self, D):
        for sym in self.idp_phy.type_names:
            D_split = list(zip(*[
                D[sym][:,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2,self.start_phy_orbs[sym][i]*2:self.start_phy_orbs[sym][i]*2+self.idp_phy.listnorbs[sym][self.idx_intorb[sym][i]]*2] 
                for i, s in enumerate(self.start_int_orbs[sym])]))

            for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
                self.interact_ansatz[ia].update_D(D_split[ita])
        
        return True

    @property
    def LAM_C(self):
        LAM_C = {sym:[[torch.eye(i*2, device=self.device) for i in self.idp_phy.listnorbs[sym]]]*int(self.atomic_number.eq(atomic_num_dict[sym]).sum()) 
               for sym in self.idp_phy.type_names}
        for sym in self.idp_phy.type_names:
            for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
                for i, into in enumerate(self.idx_intorb[sym]):
                    LAM_C[sym][ita][into] = self.interact_ansatz[ia].LAM_C[i]
                LAM_C[sym][ita] = torch.block_diag(*LAM_C[sym][ita])
            LAM_C[sym] = torch.stack(LAM_C[sym])

        return LAM_C
    
    def update_LAM_C(self, LAM_C):
        for sym in self.idp_phy.type_names:
            LAM_C_split = list(zip(*[
                LAM_C[sym][:,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2] 
                for i, s in enumerate(self.start_int_orbs[sym])]))

            for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
                self.interact_ansatz[ia].update_LAM_C(LAM_C_split[ita])

    @property
    def LAM(self):
        LAM = {sym:[[torch.eye(i*2, device=self.device) for i in self.idp_phy.listnorbs[sym]]]*int(self.atomic_number.eq(atomic_num_dict[sym]).sum()) 
               for sym in self.idp_phy.type_names}
        for sym in self.idp_phy.type_names:
            for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
                for i, into in enumerate(self.idx_intorb[sym]):
                    LAM[sym][ita][into] = self.interact_ansatz[ia].LAM[i]
                LAM[sym][ita] = torch.block_diag(*LAM[sym][ita])
            LAM[sym] = torch.stack(LAM[sym])

        return LAM
    
    def update_LAM(self, LAM):
        for sym in self.idp_phy.type_names:
            LAM_split = list(zip(*[
                LAM[sym][:,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2] 
                for i, s in enumerate(self.start_int_orbs[sym])]))

            for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
                self.interact_ansatz[ia].update_LAM(LAM_split[ita])



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

def dF(A, Hs):
    eigval, eigvec = torch.linalg.eigh(A)
    n = eigval.shape[0]
    # eigval = torch.clip(eigval, 1e-15, 1-1e-15)
    Hbar = eigvec.H @ Hs @ eigvec
    fc_eigval = eigval * (1 - eigval)
    fc_eigval = torch.sqrt(fc_eigval)
    dfc_eigval = (0.5 - eigval) / fc_eigval

    loewm = fc_eigval[:,None] - fc_eigval[None,:]
    eigdev = eigval[:, None] - eigval[None, :]
    deg_mask = eigdev.abs().lt(1e-12)
    
    idx = torch.arange(n)[:,None].repeat(1,n)
    r_idx = idx.T[deg_mask]
    l_idx = idx[deg_mask]
    loewm[l_idx, r_idx] = dfc_eigval[r_idx]
    r_idx = idx.T[~deg_mask]
    l_idx = idx[~deg_mask]
    
    loewm[l_idx, r_idx] /= eigdev[~deg_mask]
    deriv = eigvec @ (loewm[None,:,:].real * Hbar) @ eigvec.H  

    

    return deriv

def calc_lam_c(R, lam, Delta_p, D, H_list):
    DR = D @ R.T
    driv = dF(Delta_p, H_list.transpose(1,2))
    lc = -(DR[None, :,:] * driv).sum(dim=(1,2))
    lc += lc.conj()
    lc -= lam

    return lc
