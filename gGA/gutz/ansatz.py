import numpy as np
from typing import Tuple, Union, Dict
from gGA.data import OrbitalMapper
from gGA.utils.constants import atomic_num_dict_r, atomic_num_dict
import copy
from gGA.utils.tools import hermitian_basis_nspin, trans_basis_nspin
from gGA.solver import ED_solver, NQS_solver, DMRG_solver
from gGA.gutz.mixing import Linear, PDIIS
from scipy.linalg import block_diag

# single means orbital belong to single angular momentum or subspace. for example, s, p, d, d_t2g, etc.
class gGASingleOrb(object):
    def __init__(
            self, 
            norb,
            naux:int=1,
            solver: str="ED",
            nspin: int=1, # 1 for spin degenerate, 2 for collinear spin polarized, 4 for non-collinear spin polarization
            decouple_bath: bool=False, 
            natural_orbital: bool=False,
            solver_options: dict={},
            mixer_options: dict={},
            iscomplex=False,
            dtype=np.float64,
            ):
        
        super(gGASingleOrb, self).__init__()
        self.dtype = dtype
        self.solver = solver
        self.norb = norb
        self.naux = naux
        self.nauxorb = naux * norb
        self.nspin = nspin
        self.iscomplex = iscomplex

        self.phy_spinorb = self.norb*2
        self.aux_spinorb = self.naux*self.norb*2
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
        self.hermit_basis = hermitian_basis_nspin(self.aux_spinorb, self.nspin, iscomplex=iscomplex)
        self.trans_basis = trans_basis_nspin(self.aux_spinorb, self.phy_spinorb, self.nspin, iscomplex=iscomplex)
        
        self._rdm = np.zeros(int(self.nspin*self.nauxorb*(self.nauxorb+1)/2), dtype=self.dtype)
        R = np.random.rand(self.aux_spinorb//2, self.phy_spinorb//2)
        R = R + np.random.rand(*R.shape)*1j
        R = np.kron(R, np.eye(2))
        # R = torch.kron(torch.ones(4, 2), torch.eye(2))
        self.update_R(R=R)
        # self._r = torch.ones(self.nspin*self.nauxorb*self.norb, dtype=self.dtype, device=self.device) * 0.5
        # Since we know R @ R.T is a diagonal matrix, can we use this properties to accelerate convergence?

        # setup langrangian multipliers
        # LAM = torch.zeros((self.naux * self.norb * 2,), device=self.device)
        # LAM[:(naux//2)*self.norb*2] = 0.1
        # LAM[-(naux//2)*self.norb*2:] = -0.1
        # LAM = torch.diag(LAM)

        # LAM = torch.rand((self.naux * self.norb, self.naux * self.norb), device=self.device)
        LAM = np.diag(np.random.rand(self.naux * self.norb))
        LAM = np.kron(LAM-np.eye(LAM.shape[0])*(LAM.trace()/LAM.shape[0]), np.eye(2))
        # LAM = torch.kron(0.5*(LAM+LAM.T)-torch.eye(LAM.shape[0])*(LAM.trace()), torch.eye(2))
        # LAM = torch.diag(torch.tensor([1.,1.,0.5,0.5,-0.5,-0.5,-1.,-1.]))

        self.update_LAM(LAM=LAM)
        self._lamc = np.zeros(int(self.nspin*self.nauxorb*(self.nauxorb+1)/2), dtype=self.dtype)
        self._d = np.zeros(self.nspin*self.nauxorb*self.norb, dtype=self.dtype)
        self._Hemb_param_uptodate = False
        self.decouple_bath = decouple_bath
        self.natural_orbital = natural_orbital

        assert not self.decouple_bath * self.natural_orbital, "The bath cannot be decoupled when doing natural orbital transform."

        # initialize the mixing algorithm
        p0=np.concatenate([self.LAM, self.R], axis=1)

        if mixer_options.get("method") == "Linear":
            self.mixer = Linear(**mixer_options, p0=p0)
        elif mixer_options.get("method") == "PDIIS":
            self.mixer = PDIIS(**mixer_options, p0=p0)
        elif len(mixer_options)==0:
            self.mixer = None
        else:
            raise NotImplementedError("Mixer other than Linear/PDIIS have not been implemented.")

        if decouple_bath:
            assert self.solver == "DMRG" or self.solver == "NQS", "decouple bath for other method is not implemented"
        if natural_orbital:
            assert self.solver == "NQS"

        if self.solver == "ED":
            self.solver = ED_solver(
                norb=norb,
                naux=naux,
                nspin=nspin,
                iscomplex=self.iscomplex,
                decouple_bath=self.decouple_bath,
                natural_orbital=self.natural_orbital,
                **solver_options
            )

        elif self.solver == "NQS":
            self.solver = NQS_solver(
                norb=norb,
                naux=naux,
                nspin=nspin,
                decouple_bath=self.decouple_bath,
                natural_orbital=self.natural_orbital,
                iscomplex=self.iscomplex,
                **solver_options
            )

        elif self.solver == "DMRG":
            self.solver = DMRG_solver(
                norb=norb, 
                naux=naux, 
                nspin=nspin, 
                iscomplex=self.iscomplex, 
                decouple_bath=self.decouple_bath,
                natural_orbital=self.natural_orbital,
                **solver_options
                # scratch_dir="./", 
                # n_threads: int=1,
                # nupdate=4, 
                # bond_dim=250, 
                # bond_mul=2, 
                # n_sweep=20, 
                # eig_cutoff=1e-7
            )

    def update(self, t, intparam, E_fermi):
        self._lamc = calc_lam_c(self.R, self._lam, self.RDM, self.D, self.hermit_basis)
        self._Hemb_param_uptodate = False
        
        # update R, RDM
        _t, _intparam = self.get_Hemb_param(t, intparam, E_fermi)
        RDM = self.solver.solve(_t, _intparam, return_RDM=True)

        # compute R and RDM_bath
        RDM_bath = np.eye(self.aux_spinorb, dtype=self.dtype) - RDM[-self.aux_spinorb:, -self.aux_spinorb:]
        self.update_RDM(RDM=RDM_bath) # symmetrization first
        RDM_bath = self.RDM
        
        A = RDM_bath @ (np.eye(self.aux_spinorb, dtype=self.dtype)-RDM_bath)
        A = symsqrt(A).T
        B = (RDM[:self.phy_spinorb, -self.aux_spinorb:].T[None,:,:] * self.trans_basis).sum(axis=(1,2))
        B = 0.5 * (B + B.conj()).real
        B = (B[:,None,None] * self.trans_basis).sum(axis=0).conj()
        R = np.linalg.solve(a=A, b=B).reshape(self.aux_spinorb, self.phy_spinorb)

        self.update_R(R=R)
        self._lam = calc_lam_c(self.R, self._lamc, RDM_bath, self.D, self.hermit_basis)

        if self.mixer is not None:
            p = self.mixer.update(np.concatenate([self.LAM, self.R], axis=1))
            self.update_LAM(p[:,:self.aux_spinorb])
            self.update_R(p[:,self.aux_spinorb:])
        
        return True
    
    def fix_gauge(self):
        # fix gauge of lambda and R
        R, LAM = self.R, self.LAM
        eigval, eigvec = np.linalg.eigh(LAM)
        LAM = np.diag(eigval)
        R = eigvec.T @ R

        mat = np.ones_like(R[:,0])
        mat[R[:,0] < 0] = -1
        mat = np.diag(mat)
        R = mat @ R
        LAM = mat @ LAM @ mat.T

        idx = np.argsort(R[:,0])[::-1]
        mat = np.eye(R.shape[0])[idx]
        LAM = mat @ LAM @ mat.T
        R = mat @ R

        self.update_LAM(LAM=LAM)
        self.update_R(R=R)

        self._Hemb_uptodate = False

        return True


    @property
    def RDM(self):
        return (self.hermit_basis * self._rdm[:,None,None]).sum(axis=0)
    
    def update_RDM(self, RDM):
        self._rdm = (RDM[None,...] * self.hermit_basis.conj()).sum(axis=(1,2)).real
        self._Hemb_uptodate = False
        return True
    
    @property
    def R(self):
        return (self.trans_basis * self._r[:, None, None]).sum(0).conj()

    def update_R(self, R):
        self._r = (R[None,...] * self.trans_basis).sum(axis=(1,2))
        self._r = 0.5 * (self._r + self._r.conj()).real
        self._Hemb_uptodate = False
        return True

    @property
    def D(self):
        return (self.trans_basis * self._d[:, None, None]).sum(0).conj()

    def update_D(self, D):
        self._d = (D[None,...] * self.trans_basis).sum(axis=(1,2))
        self._d = 0.5 * (self._d + self._d.conj()).real
        self._Hemb_uptodate = False
        return True

    @property
    def LAM_C(self):
        return (self.hermit_basis * self._lamc[:,None,None]).sum(axis=0)
    
    def update_LAM_C(self, LAM_C):
        self._lamc = (self.hermit_basis.conj() * LAM_C[None,...]).sum(axis=(1,2)).real
        self._Hemb_uptodate = False
        return True

    @property
    def LAM(self):
        return (self.hermit_basis * self._lam[:,None,None]).sum(axis=0)
    
    @property
    def E(self):
        return self.solver.E
    
    @property
    def docc(self):
        return self.solver.docc
    
    @property
    def fRDM(self):
        return self.solver.RDM
    
    def update_LAM(self, LAM):
        self._lam = (self.hermit_basis.conj() * LAM[None,...]).sum(axis=(1,2)).real
        self._Hemb_uptodate = False
        return True


    def get_Hemb_param(self, t, intparam, E_fermi):
        if not self._Hemb_param_uptodate or abs(self._t - t).max() > 1e-6 or not intparam == self._intparam:
            return self._calc_Hemb_param(t, intparam, E_fermi)

        else:
            return self._t, self._intparam


    def _calc_Hemb_param(self, t, intparam, E_fermi):
        # construct the embedding Hamiltonian
        # assert self.decouple_bath + (not self.natural_orbital), "Not support natural orbital representation of bath when bath is not decoupled."

        # constructing the kinetical part T
        T = np.zeros((self.aux_spinorb+self.phy_spinorb, self.aux_spinorb+self.phy_spinorb), dtype=self.dtype)
        if self.iscomplex:
            T = T + 0j
        if isinstance(t, np.ndarray):
            if t.shape[0] == self.norb: # spin degenerate
                assert self.nspin == 1
                T[:self.norb*2, :self.norb*2] = block_diag(t,t).reshape(2,self.norb,2,-1).transpose(1,0,3,2).reshape(self.norb*2,-1)
            else:
                assert t.shape[0] == self.norb*2, "the hopping t's shape does not match either spin denegrate case or spin case"
                T[:self.norb*2, :self.norb*2] = t
        else:
            assert isinstance(t, (int, float))
            T[:self.norb*2, self.norb*2:] = np.eye(self.norb*2, dtype=self.dtype) * t
        
        T[:self.norb*2, :self.norb*2] -= E_fermi * np.eye(self.norb*2, dtype=self.dtype)
        T[:self.phy_spinorb, self.phy_spinorb:] = self.D.T
        T[self.phy_spinorb:, :self.phy_spinorb] = self.D.conj()
        T[self.phy_spinorb:, self.phy_spinorb:] = -self.LAM_C
        self._t = T.copy()
        self._intparam = intparam

        # if self.natural_orbital:
        #     # TODO, do the transformation latter
        #     raise NotImplementedError

        self._Hemb_param_uptodate = True

        return T, intparam

class gGAMultiOrb(object):
    def __init__(
            self, 
            norbs,
            naux:int=1,
            nspin:int=1,
            solver: str="ED",
            decouple_bath: bool=False,
            natural_orbital: bool=False,
            solver_options: dict={},
            mixer_options: dict={},
            iscomplex=False,
            ):
        
        super(gGAMultiOrb, self).__init__()
        self.norbs = norbs
        self.naux = naux
        self.nspin = nspin
        self.solver = solver
        self.orbcount = sum(norbs)
        self.singleOrbs = [
            gGASingleOrb(
                norb,
                naux,
                nspin=nspin, 
                solver=solver, 
                decouple_bath=decouple_bath, 
                natural_orbital=natural_orbital, 
                solver_options=solver_options,
                mixer_options=mixer_options,
                iscomplex=iscomplex,
                ) for i, norb in enumerate(norbs)
                ]

    def update(self, t, intparams, E_fermi):
        # LAM_mo, Lambda lagrangian multipliers of multi-orbital in matrix form, should have shape (sum(norbs)*naux*2, sum(norbs)*naux*2)
        for iso, singleOrb in enumerate(self.singleOrbs):
            t_so = t[iso] # [aux_spinorb, aux_spinorb]
            singleOrb.update(t_so, intparams[iso], E_fermi)

        return True
    
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
    
    @property
    def E(self):
        return sum([singleOrb.E for singleOrb in self.singleOrbs])
    
    @property
    def docc(self):
        return [singleOrb.docc for singleOrb in self.singleOrbs]
    
    @property
    def fRDM(self):
        return [singleOrb.fRDM for singleOrb in self.singleOrbs]

class gGAtomic(object):
    def __init__(
            self, 
            basis: dict, 
            atomic_number: np.ndarray, 
            idx_intorb: Dict[str, list], 
            naux: int, 
            nspin: int,
            solver: str="ED",
            decouple_bath: bool=False,
            natural_orbital: bool=False,
            solver_options: dict={},
            mixer_options: dict={},
            iscomplex=False,
            dtype=np.float64,
        ):

        self.basis = basis
        self.atomic_number = atomic_number
        self.naux = naux
        self.nspin = nspin
        self.solver = solver
        self.spin_deg = self.nspin <= 1
        self.idp_phy = OrbitalMapper(basis=basis, spin_deg=self.spin_deg)
        self.idx_intorb = idx_intorb
        self.solver = solver
        # do some orbital statistics
        self.interact_ansatz = []
        self.interacting_atoms = []
        self.interacting_idx = [] # no.x atom of this interacting atoms in all of its species

        which_sym = {sym: 0 for sym in self.idx_intorb}
        for ia, an in enumerate(atomic_number):
            sym = atomic_num_dict_r[int(atomic_number[ia])]
            if sym in idx_intorb.keys():
                self.interacting_atoms.append(ia)
                self.interacting_idx.append(which_sym[sym])
                which_sym[sym] = which_sym[sym] + 1
                intorb_a = [self.idp_phy.listnorbs[sym][i] for i in idx_intorb[atomic_num_dict_r[int(atomic_number[ia])]]]
                self.interact_ansatz.append(
                    gGAMultiOrb(
                        norbs=intorb_a,
                        naux=naux, 
                        nspin=nspin, 
                        solver=solver,
                        natural_orbital=natural_orbital,
                        decouple_bath=decouple_bath,
                        solver_options=solver_options,
                        mixer_options=mixer_options,
                        iscomplex=iscomplex,
                        )
                    ) # TODO: The most computational demanding part, would be perfect if it is parallizable

        # generate the idp for gost system
        nauxorbs = copy.deepcopy(self.idp_phy.listnorbs) # {'C': [1, 3], 'Si': [1, 1, 3, 3, 5]}
        
        for sym in self.idx_intorb.keys():
            for i in self.idx_intorb[sym]:
                nauxorbs[sym][i] = nauxorbs[sym][i] * naux
        self.nauxorbs = nauxorbs

        start_int_orbs = {sym:[] for sym in self.idx_intorb.keys()}
        start_phy_orbs = {sym:[] for sym in self.idx_intorb.keys()}
        for sym, ii in self.idx_intorb.items():
            for i in ii:
                start_int_orbs[sym].append(sum(nauxorbs[sym][:i])*2)
                start_phy_orbs[sym].append(sum(self.idp_phy.listnorbs[sym][:i]))
        self.start_int_orbs = start_int_orbs
        self.start_phy_orbs = start_phy_orbs

    def update(self, t, intparams, E_fermi):
        # LAM should have a stacked matrix format, with shape [natoms, sum(nintorbs)*naux*2, sum(nintorbs)*naux*2]
        for sym, param in intparams.items():
            assert len(param) == len(self.idx_intorb[sym]), "Hint_params should have the same length as idx_intorb"

        for idx, aid in enumerate(self.interacting_atoms):
            sym = atomic_num_dict_r[int(self.atomic_number[aid])]
            ita = self.interacting_idx[idx]
            # ita for the ith atom in sym type, ia is its id in atomic number
            t_a = t[sym][ita]
            assert t_a.shape[1] == sum(self.idp_phy.listnorbs[sym])*2, "Shape of t is not correct!"
            t_a = [t_a[s*2:s*2+self.idp_phy.listnorbs[sym][self.idx_intorb[sym][i]]*2,s*2:s*2+self.idp_phy.listnorbs[sym][self.idx_intorb[sym][i]]*2] for i, s in enumerate(self.start_phy_orbs[sym])]
            self.interact_ansatz[idx].update(t_a, intparams[sym], E_fermi)
                # update the R, RDM for each atomic system

        return True

    def fix_gauge(self):
        for i in range(len(self.atomic_number)):
            self.interact_ansatz[i].fix_gauge()

        return True

    @property
    def R(self):
        R = {sym:[[np.eye(i*2) for i in self.idp_phy.listnorbs[sym]]]*int((self.atomic_number == atomic_num_dict[sym]).sum()) 
               for sym in self.idx_intorb.keys()}
        
        for idx, aid in enumerate(self.interacting_atoms):
            sym = atomic_num_dict_r[int(self.atomic_number[aid])]
            ita = self.interacting_idx[idx]
            for i, into in enumerate(self.idx_intorb[sym]):
                R[sym][ita][into] = self.interact_ansatz[idx].R[i]
            R[sym][ita] = block_diag(*R[sym][ita])
        for sym in R:
            R[sym] = np.stack(R[sym])

        return R
    
    def update_R(self, R):
        R_split = {}
        for sym in self.idx_intorb.keys():
            R_split[sym] = list(zip(*[
                R[sym][:,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2,self.start_phy_orbs[sym][i]*2:self.start_phy_orbs[sym][i]*2+self.idp_phy.listnorbs[sym][self.idx_intorb[sym][i]]*2] 
                for i, s in enumerate(self.start_int_orbs[sym])]))
        
        for idx, aid in enumerate(self.interacting_atoms):
            ita = self.interacting_idx[idx]
            sym = atomic_num_dict_r[int(self.atomic_number[aid])]
            self.interact_ansatz[idx].update_R(R_split[sym][ita])
        
        return True

    @property
    def RDM(self):
        RDM = {sym:[[np.eye(i*2)*(1/(i*2)) for i in self.idp_phy.listnorbs[sym]]]*int((self.atomic_number == atomic_num_dict[sym]).sum()) 
               for sym in self.idx_intorb.keys()}
        
        for idx, aid in enumerate(self.interacting_atoms):
            sym = atomic_num_dict_r[int(self.atomic_number[aid])]
            ita = self.interacting_idx[idx]
            for i, into in enumerate(self.idx_intorb[sym]):
                RDM[sym][ita][into] = self.interact_ansatz[idx].RDM[i]
            RDM[sym][ita] = block_diag(*RDM[sym][ita])
        for sym in RDM:
            RDM[sym] = np.stack(RDM[sym])

        # for sym in self.idp_phy.type_names:
        #     for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
        #         for i, into in enumerate(self.idx_intorb[sym]):
        #             RDM[sym][ita][into] = self.interact_ansatz[ia].RDM[i]
        #         RDM[sym][ita] = torch.block_diag(*RDM[sym][ita])
        #     RDM[sym] = torch.stack(RDM[sym])

        return RDM
    
    @property
    def fRDM(self): #TODO: this is not right

        RDM = {sym:[[np.eye(i*2)*(1/(i*2)) for i in self.idp_phy.listnorbs[sym]]]*int((self.atomic_number == atomic_num_dict[sym]).sum()) 
               for sym in self.idx_intorb.keys()}
        
        for idx, aid in enumerate(self.interacting_atoms):
            sym = atomic_num_dict_r[int(self.atomic_number[aid])]
            ita = self.interacting_idx[idx]
            for i, into in enumerate(self.idx_intorb[sym]):
                RDM[sym][ita][into] = self.interact_ansatz[idx].fRDM[i]
            RDM[sym][ita] = block_diag(*RDM[sym][ita])
        for sym in RDM:
            RDM[sym] = np.stack(RDM[sym])

        # RDM = {sym:[[torch.eye(i*2, device=self.device)*(1/(i*2)) for i in self.idp_phy.listnorbs[sym]]]*int(self.atomic_number.eq(atomic_num_dict[sym]).sum()) 
        #        for sym in self.idp_phy.type_names}
        # for sym in self.idp_phy.type_names:
        #     for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
        #         for i, into in enumerate(self.idx_intorb[sym]):
        #             RDM[sym][ita][into] = self.interact_ansatz[ia].fRDM[i]
        #         RDM[sym][ita] = torch.block_diag(*RDM[sym][ita])
        #     RDM[sym] = torch.stack(RDM[sym])

        return RDM
    
    @property
    def E(self):
        return sum([ansatz.E for ansatz in self.interact_ansatz])
    
    @property
    def docc(self):
        DOCC = {sym:[[np.zeros(i) for i in self.idp_phy.listnorbs[sym]]]*int((self.atomic_number == atomic_num_dict[sym]).sum()) 
               for sym in self.idx_intorb.keys()}
        
        for idx, aid in enumerate(self.interacting_atoms):
            sym = atomic_num_dict_r[int(self.atomic_number[aid])]
            ita = self.interacting_idx[idx]
            for i, into in enumerate(self.idx_intorb[sym]):
                DOCC[sym][ita][into] = self.interact_ansatz[idx].docc[i]
            DOCC[sym][ita] = np.concatenate(DOCC[sym][ita])
        for sym in DOCC:
            DOCC[sym] = np.stack(DOCC[sym])
        
        # for sym in self.idp_phy.type_names:
        #     for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
        #         for i, into in enumerate(self.idx_intorb[sym]):
        #             DOCC[sym][ita][into] = self.interact_ansatz[ia].docc[i]
        #         DOCC[sym][ita] = torch.cat(DOCC[sym][ita])
        #     DOCC[sym] = torch.stack(DOCC[sym])
        return DOCC

    def update_RDM(self, RDM):
        # RDM should have the same shape as the property RDM

        RDM_split = {}
        for sym in self.idx_intorb.keys():
            RDM_split[sym] = list(zip(*[
                RDM[sym][:,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2] 
                for i, s in enumerate(self.start_int_orbs[sym])]))
        
        for idx, aid in enumerate(self.interacting_atoms):
            ita = self.interacting_idx[idx]
            sym = atomic_num_dict_r[int(self.atomic_number[aid])]
            self.interact_ansatz[idx].update_RDM(RDM_split[sym][ita])

        # for sym in self.idp_phy.type_names:
        #     RDM_split = list(zip(*[
        #         RDM[sym][:,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2] 
        #         for i, s in enumerate(self.start_int_orbs[sym])]))

        #     for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
        #         self.interact_ansatz[ia].update_RDM(RDM_split[ita])
        
        return True


    @property
    def D(self):
        D = {sym:[[np.eye(i*2) for i in self.idp_phy.listnorbs[sym]]]*int((self.atomic_number == atomic_num_dict[sym]).sum()) 
               for sym in self.idx_intorb.keys()}
        
        for idx, aid in enumerate(self.interacting_atoms):
            sym = atomic_num_dict_r[int(self.atomic_number[aid])]
            ita = self.interacting_idx[idx]
            for i, into in enumerate(self.idx_intorb[sym]):
                D[sym][ita][into] = self.interact_ansatz[idx].D[i]
            D[sym][ita] = block_diag(*D[sym][ita])
        for sym in D:
            D[sym] = np.stack(D[sym])

        # D = {sym:[[torch.eye(i*2, device=self.device) for i in self.idp_phy.listnorbs[sym]]]*int(self.atomic_number.eq(atomic_num_dict[sym]).sum()) 
        #        for sym in self.idp_phy.type_names}
        # for sym in self.idp_phy.type_names:
        #     for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
        #         for i, into in enumerate(self.idx_intorb[sym]):
        #             D[sym][ita][into] = self.interact_ansatz[ia].D[i]
        #         D[sym][ita] = torch.block_diag(*D[sym][ita])
        #     D[sym] = torch.stack(D[sym])

        return D

    def update_D(self, D):

        D_split = {}
        for sym in self.idx_intorb.keys():
            D_split[sym] = list(zip(*[
                D[sym][:,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2,self.start_phy_orbs[sym][i]*2:self.start_phy_orbs[sym][i]*2+self.idp_phy.listnorbs[sym][self.idx_intorb[sym][i]]*2] 
                for i, s in enumerate(self.start_int_orbs[sym])]))
        
        for idx, aid in enumerate(self.interacting_atoms):
            ita = self.interacting_idx[idx]
            sym = atomic_num_dict_r[int(self.atomic_number[aid])]
            self.interact_ansatz[idx].update_D(D_split[sym][ita])

        # for sym in self.idp_phy.type_names:
        #     D_split = list(zip(*[
        #         D[sym][:,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2,self.start_phy_orbs[sym][i]*2:self.start_phy_orbs[sym][i]*2+self.idp_phy.listnorbs[sym][self.idx_intorb[sym][i]]*2] 
        #         for i, s in enumerate(self.start_int_orbs[sym])]))

        #     for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
        #         self.interact_ansatz[ia].update_D(D_split[ita])
        
        return True

    @property
    def LAM_C(self):

        LAM_C = {sym:[[np.zeros((i*2, i*2)) for i in self.idp_phy.listnorbs[sym]]]*int((self.atomic_number == atomic_num_dict[sym]).sum()) 
               for sym in self.idx_intorb.keys()}
        
        for idx, aid in enumerate(self.interacting_atoms):
            sym = atomic_num_dict_r[int(self.atomic_number[aid])]
            ita = self.interacting_idx[idx]
            for i, into in enumerate(self.idx_intorb[sym]):
                LAM_C[sym][ita][into] = self.interact_ansatz[idx].LAM_C[i]
            LAM_C[sym][ita] = block_diag(*LAM_C[sym][ita])
        for sym in LAM_C:
            LAM_C[sym] = np.stack(LAM_C[sym])

        # LAM_C = {sym:[[torch.zeros((i*2, i*2), device=self.device) for i in self.idp_phy.listnorbs[sym]]]*int(self.atomic_number.eq(atomic_num_dict[sym]).sum()) 
        #        for sym in self.idp_phy.type_names}
        # for sym in self.idp_phy.type_names:
        #     for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
        #         for i, into in enumerate(self.idx_intorb[sym]):
        #             LAM_C[sym][ita][into] = self.interact_ansatz[ia].LAM_C[i]
        #         LAM_C[sym][ita] = torch.block_diag(*LAM_C[sym][ita])
        #     LAM_C[sym] = torch.stack(LAM_C[sym])

        return LAM_C
    
    def update_LAM_C(self, LAM_C):
        LAM_C_split = {}
        for sym in self.idx_intorb.keys():
            LAM_C_split[sym] = list(zip(*[
                LAM_C[sym][:,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2] 
                for i, s in enumerate(self.start_int_orbs[sym])]))
        
        for idx, aid in enumerate(self.interacting_atoms):
            ita = self.interacting_idx[idx]
            sym = atomic_num_dict_r[int(self.atomic_number[aid])]
            self.interact_ansatz[idx].update_LAM_C(LAM_C_split[sym][ita])

        # for sym in self.idp_phy.type_names:
        #     LAM_C_split = list(zip(*[
        #         LAM_C[sym][:,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2] 
        #         for i, s in enumerate(self.start_int_orbs[sym])]))

        #     for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
        #         self.interact_ansatz[ia].update_LAM_C(LAM_C_split[ita])

    @property
    def LAM(self):
        LAM = {sym:[[np.zeros((i*2, i*2)) for i in self.idp_phy.listnorbs[sym]]]*int((self.atomic_number == atomic_num_dict[sym]).sum()) 
               for sym in self.idx_intorb.keys()}
        
        for idx, aid in enumerate(self.interacting_atoms):
            sym = atomic_num_dict_r[int(self.atomic_number[aid])]
            ita = self.interacting_idx[idx]
            for i, into in enumerate(self.idx_intorb[sym]):
                LAM[sym][ita][into] = self.interact_ansatz[idx].LAM[i]
            LAM[sym][ita] = block_diag(*LAM[sym][ita])
        for sym in LAM:
            LAM[sym] = np.stack(LAM[sym])

        # LAM = {sym:[[torch.zeros((i*2, i*2), device=self.device) for i in self.idp_phy.listnorbs[sym]]]*int(self.atomic_number.eq(atomic_num_dict[sym]).sum()) 
        #        for sym in self.idp_phy.type_names}
        # for sym in self.idp_phy.type_names:
        #     for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
        #         for i, into in enumerate(self.idx_intorb[sym]):
        #             LAM[sym][ita][into] = self.interact_ansatz[ia].LAM[i]
        #         LAM[sym][ita] = torch.block_diag(*LAM[sym][ita])
        #     LAM[sym] = torch.stack(LAM[sym])

        return LAM
    
    def update_LAM(self, LAM):
        LAM_split = {}
        for sym in self.idx_intorb.keys():
            LAM_split[sym] = list(zip(*[
                LAM[sym][:,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2] 
                for i, s in enumerate(self.start_int_orbs[sym])]))
        
        for idx, aid in enumerate(self.interacting_atoms):
            ita = self.interacting_idx[idx]
            sym = atomic_num_dict_r[int(self.atomic_number[aid])]
            self.interact_ansatz[idx].update_LAM(LAM_split[sym][ita])

        # for sym in self.idp_phy.type_names:
        #     LAM_split = list(zip(*[
        #         LAM[sym][:,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2,s:s+self.nauxorbs[sym][self.idx_intorb[sym][i]]*2] 
        #         for i, s in enumerate(self.start_int_orbs[sym])]))

        #     for ita, ia in enumerate(torch.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
        #         self.interact_ansatz[ia].update_LAM(LAM_split[ita])



def symsqrt(matrix): # this may returns nan grade when the eigenvalue of the matrix is very degenerated.
    """Compute the square root of a positive definite matrix."""
    # _, s, v = safeSVD(matrix)
    _, s, v = np.linalg.svd(matrix)
    v = v.conj().T

    good = s > s.max(axis=-1, keepdims=True) * s.shape[-1] * np.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.shape[-1]:
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, np.zeros((), dtype=s.dtype))
    shape = list(s.shape[:-1]) + [1] + [s.shape[-1]]
    return (v * np.sqrt(s).reshape(shape)) @ v.conj().swapaxes(-1,-2)

def dF(A, Hs):
    eigval, eigvec = np.linalg.eigh(A)
    n = eigval.shape[0]
    # eigval = torch.clip(eigval, 1e-15, 1-1e-15)
    Hbar = eigvec.conj().T @ Hs @ eigvec
    fc_eigval = eigval * (1 - eigval)
    fc_eigval = np.sqrt(fc_eigval)
    dfc_eigval = (0.5 - eigval) / fc_eigval

    loewm = fc_eigval[:,None] - fc_eigval[None,:]
    eigdev = eigval[:, None] - eigval[None, :]
    deg_mask = np.abs(eigdev) < 1e-12
    
    idx = np.arange(n)[:,None].repeat(n, axis=1)
    r_idx = idx.T[deg_mask]
    l_idx = idx[deg_mask]
    loewm[l_idx, r_idx] = dfc_eigval[r_idx]
    r_idx = idx.T[~deg_mask]
    l_idx = idx[~deg_mask]
    
    loewm[l_idx, r_idx] /= eigdev[~deg_mask]
    deriv = eigvec @ (loewm[None,:,:].real * Hbar) @ eigvec.conj().T

    return deriv

def calc_lam_c(R, lam, Delta_p, D, H_list):
    DR = D @ R.T
    driv = dF(Delta_p, H_list.swapaxes(1,2))
    lc = -(DR.T[None, :,:] * driv).sum(axis=(1,2))
    lc += lc.conj()
    lc -= lam

    return lc.real
