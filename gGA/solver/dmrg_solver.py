# Refer to the usage of block2: https://block2.readthedocs.io/en/latest/tutorial/hubbard.html
import numpy as np
try:
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
except:
    print("The block2 is not installed. One should not use DMRG solver.")
from quspin.operators._make_hamiltonian import _consolidate_static
import numpy as np
import copy
from typing import Dict
from gGA.operator import Slater_Kanamori, create_d, annihilate_d, create_u, annihilate_u, number_d, number_u, S_z, S_m, S_p

class DMRG_solver(object):
    def __init__(
            self, 
            norb, 
            naux, 
            nspin,
            decouple_bath: bool=False, 
            natural_orbital: bool=False,  
            iscomplex=False, 
            scratch_dir="./dmrg_tmp", 
            n_threads: int=1,
            kBT: float=0.025,
            mutol: float=1e-4,
            nupdate=4, 
            bond_dim=250, 
            bond_mul=2, 
            n_sweep=20, 
            eig_cutoff=1e-7
            ) -> None:
        
        self.norb = norb
        self.naux = naux
        self.nspin = nspin
        self.iscomplex = iscomplex

        self.nupdate = nupdate
        self.bond_dim = bond_dim
        self.bond_mul = bond_mul
        self.n_sweep = n_sweep
        self.eig_cutoff = eig_cutoff
        self.decouple_bath = decouple_bath
        self.natural_orbital = natural_orbital

        if self.nspin == 1:
            self.driver = DMRGDriver(scratch=scratch_dir, symm_type=SymmetryTypes.SZ, n_threads=n_threads)
            self.driver.initialize_system(n_sites=self.norb*(self.naux+1), n_elec=self.norb*(self.naux+1), spin=0)
            # Nparticle = [(self.norb*(self.naux+1)-i,i) for i in range(0,self.norb*(self.naux+1)+1)]
            self.Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]
        else:
            self.driver = DMRGDriver(scratch=scratch_dir, symm_type=SymmetryTypes.SGF, n_threads=n_threads)
            self.driver.initialize_system(n_sites=self.norb*(self.naux+1)*2, n_elec=self.norb*(self.naux+1))
            # self.Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]
            self.Nparticle = [(self.norb*(self.naux+1)-i,i) for i in range(0,self.norb*(self.naux+1)+1)]

        self._t = 0.
        self._intparam = {}

        self.op_map = {
            "+-": np.array([
                ["CD", "Cd"],
                ["cD", "cd"],
            ]),
            "-+": np.array([
                ["DC", "Dc"],
                ["dC", "dc"],
            ]),
            "nn": np.array(["CDCD","err", "err", "CDcd", 
                              "err", "err", "err", "err", 
                              "err", "err", "err", "err", 
                              "cdCD", "err", "err", "cdcd"]).reshape(2,2,2,2),

            "+-+-": np.array(["CDCD","CDCd", "CDcD", "CDcd", 
                              "CdCD", "CdCd", "CdcD", "Cdcd", 
                              "cDCD", "cDCd", "cDcD", "cDcd", 
                              "cdCD", "cdCd", "cdcD", "cdcd"]).reshape(2,2,2,2),

            "++--": np.array(["CCDD", "CCDd", "CCdD", "CCdd", 
                              "CcDD", "CcDd", "CcdD", "Ccdd", 
                              "cCDD", "cCDd", "cCdD", "cCdd", 
                              "ccDD", "ccDd", "ccdD", "ccdd"]).reshape(2,2,2,2),
            "n": np.array([
                ["CD", "exx"],
                ["exx", "CD"]
                
            ])
        }

    def cal_decoupled_bath(self, T: np.array):
        nsite = self.norb * (1+self.naux)
        dc_T = T.reshape(nsite, 2, nsite, 2).copy()
        if self.nspin == 1:
            assert np.abs(dc_T[:,0,:,0] - dc_T[:,1,:,1]).max() < 1e-7
            _, eigvec = np.linalg.eigh(dc_T[:,0,:,0][nsite:,nsite:])
            temp_T = dc_T[:,0,:,0]
            temp_T[nsite:] = eigvec.conj().T @ temp_T[nsite:]
            temp_T[:,nsite:] = temp_T[:, nsite:] @ eigvec
            dc_T[:,0,:,0] = dc_T[:,1,:,1] = temp_T

            self.transmat = eigvec
        
        elif self.nspin == 2:
            _, eigvecup = np.linalg.eigh(dc_T[:,0,:,0][nsite:,nsite:])
            _, eigvecdown = np.linalg.eigh(dc_T[:,1,:,1][nsite:,nsite:])
            temp_T = dc_T[:,0,:,0]
            temp_T[nsite:] = eigvecup.conj().T @ temp_T[nsite:]
            temp_T[:,nsite:] = temp_T[:, nsite:] @ eigvecup
            dc_T[:,0,:,0] = temp_T
            temp_T = dc_T[:,1,:,1].copy()
            temp_T[nsite:] = eigvecdown.conj().T @ temp_T[nsite:]
            temp_T[:,nsite:] = temp_T[:, nsite:] @ eigvecdown
            dc_T[:,1,:,1] = temp_T

            self.transmat = [eigvecup, eigvecdown]
        
        else:
            dc_T = dc_T.reshape(nsite*2, nsite*2)
            _, eigvec = np.linalg.eigh(dc_T[nsite*2:,nsite*2:])
            dc_T[nsite*2:] = eigvec.conj().T @ dc_T[nsite*2:]
            dc_T[:,nsite*2:] = dc_T[:,nsite*2:] @ eigvec

            self.transmat = eigvec
        
        return dc_T.reshape(nsite*2, nsite*2)

    def recover_decoupled_bath(self, T: np.array):
        nsite = self.norb * (1+self.naux)
        rc_T = T.reshape(nsite,2,nsite,2).copy()
        if self.nspin == 1:
            assert np.abs(rc_T[:,0,:,0] - rc_T[:,1,:,1]).max() < 1e-7
            temp_T = rc_T[:,0,:,0]
            temp_T[nsite:] = self.transmat @ temp_T[nsite:]
            temp_T[:,nsite:] = temp_T[:, nsite:] @ self.transmat.conj().T
            rc_T[:,0,:,0] = rc_T[:,1,:,1] = temp_T

        if self.nspin == 2:
            temp_T = rc_T[:,0,:,0]
            temp_T[nsite:] = self.transmat[0] @ temp_T[nsite:]
            temp_T[:,nsite:] = temp_T[:, nsite:] @ self.transmat[0].conj().T
            rc_T[:,0,:,0] = temp_T
            temp_T = rc_T[:,1,:,1].copy()
            temp_T[nsite:] = self.transmat[1] @ temp_T[nsite:]
            temp_T[:,nsite:] = temp_T[:, nsite:] @ self.transmat[1].conj().T
            rc_T[:,1,:,1] = temp_T

        if self.nspin == 4:
            rc_T = rc_T.reshape(nsite*2, nsite*2)
            rc_T[nsite*2:] = self.transmat @ rc_T[nsite*2:]
            rc_T[:,nsite*2:] = rc_T[:,nsite*2:] @ self.transmat.conj().T
        
        return rc_T.reshape(nsite*2, nsite*2)

    def _construct_Hemb(self, T: np.ndarray, intparam: Dict[str, float]):
        # construct the embedding Hamiltonian
        if self.decouple_bath:
            assert T.shape[0] == (self.naux+1) * self.norb * 2
            # The first (self.norb, self.norb)*2 is the impurity, which is keeped unchange.
            # We do diagonalization of the other block
            T = self.cal_decoupled_bath(T=T)

        intparam = copy.deepcopy(intparam)
        intparam["t"] = T.copy()
        self._t = T
        self._intparam = intparam

        oplist = _consolidate_static( # to combine reducible op
                Slater_Kanamori(
                nsites=self.norb*(self.naux+1),
                n_noninteracting=self.norb*self.naux,
                **intparam
            ).op_list
        )
        # we should notice that the spinorbital are not adjacent in the quspin hamiltonian, 
        # so properties computed from this need to be transformed.
        b = self.driver.expr_builder()

        # SZ and SGF (w.r.t. nspin=1,2/4) use different oplist definition, try to differentiate these two
            # C,D,c,d for up_create, up_anni, down_create, down_anni, SGF mode only have CD
        

        b2oplist = []
        for op in oplist:
            str, idx, t = op # str could be "+-", "-+", "nn", "+-+-", "++--", "n"
            if str == "nn":
                idx = [idx[0],idx[0],idx[1],idx[1]]
            elif str == "n":
                idx = [idx[0], idx[0]]
            
            if self.nspin == 1:
                spin, idx = list(map(lambda x: x // (self.naux+1)*self.norb, idx)), list(map(lambda x: x % (self.naux+1)*self.norb, idx))
            else:
                spin = [0] * len(idx) # spin: 0 for up (C,D), 1 for down [c,d]
            
            spin = tuple(spin)
            b2str = self.op_map[str][spin]
            b2oplist.append((b2str, idx, t))

            b.add_term(*b2oplist[-1])
        
        self._Hemb = self.driver.get_mpo(b.finalize(), iprint=2)

        return self._Hemb
    
    def get_Hemb(self, T, intparam):
        if np.abs(self._t - T).max() > 1e-8 or self._intparam != intparam:
            return self._construct_Hemb(T, intparam)
        else:
            return self._Hemb
        
    def solve(self, T, intparam, return_RDM: bool=False):
        RDM = None
        Hemb = self.get_Hemb(T=T, intparam=intparam)
        nsites = self.norb*(self.naux+1)
        # Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]

        self.ket = self.driver.get_random_mps(tag="KET", bond_dim=self.bond_dim, nroots=1)
        bond_dims = [self.bond_dim] * self.nupdate + [self.bond_mul*self.bond_dim] * self.nupdate
        noises = [1e-4] * self.nupdate + [1e-5] * self.nupdate + [0]
        thrds = [1e-10] * 2*self.nupdate
        self.driver.dmrg(Hemb, self.ket, n_sweeps=self.n_sweep, bond_dims=bond_dims, noises=noises, thrds=thrds, cutoff=self.eig_cutoff, iprint=1)

        if return_RDM:
            RDM = self.driver.get_1pdm(self.ket)
            if self.nspin == 1:
                RDM = np.kron(0.5 * (RDM[0] + RDM[1]), np.eye(2))
            else:
                # SGF mode, the first
                RDM = RDM.reshape(2, nsites, 2, nsites).transpose(1,0,3,2).reshape(nsites*2, nsites*2)
            
            if self.decouple_bath:
                RDM = self.recover_decoupled_bath(RDM)
                
        return RDM
    
    def cal_E(self, vec):
        intparam = copy.deepcopy(self._intparam)
        intparam["t"][self.norb*2:] = 0.
        intparam["t"][:, self.norb*2:] = 0.

        oplist = _consolidate_static( # to combine reducible op
                Slater_Kanamori(
                nsites=self.norb*(self.naux+1),
                n_noninteracting=self.norb*self.naux,
                **intparam
            ).op_list
        )

        b = self.driver.expr_builder()
        b2oplist = []
        for op in oplist:
            str, idx, t = op # str could be "+-", "-+", "nn", "+-+-", "++--", "n"
            if self.nspin == 1:
                spin, idx = map(lambda x: x // (self.naux+1)*self.norb, idx), map(lambda x: x % (self.naux+1)*self.norb, idx)
            else:
                spin = [0] * len(idx) # spin: 0 for up (C,D), 1 for down [c,d]
            
            b2str = self.op_map[str][spin]
            b2oplist.append((b2str, idx, t))

            b.add_term(*b2oplist[-1])
        
        Hemb = self.driver.get_mpo(b.finalize(), iprint=2)

        E = self.driver.expectation(vec, Hemb, vec)

        return E
    
    def cal_S2(self, vec):
        nsites = (self.naux + 1) * self.norb

        S2 = np.zeros(self.norb)
        for i in range(self.norb):
            oplist = _consolidate_static( # to combine reducible op
                S_m(nsites, i) * S_p(nsites, i) + S_z(nsites, i) * S_z(nsites, i) + S_z(nsites, i)
            )
            b = self.driver.expr_builder()
            b2oplist = []
            for op in oplist:
                str, idx, t = op # str could be "+-", "-+", "nn", "+-+-", "++--", "n"
                if self.nspin == 1:
                    spin, idx = map(lambda x: x // nsites, idx), map(lambda x: x % nsites, idx)
                else:
                    spin = [0] * len(idx) # spin: 0 for up (C,D), 1 for down [c,d]
                
                b2str = self.op_map[str][spin]
                b2oplist.append((b2str, idx, t))

                b.add_term(*b2oplist[-1])
            
            S2op = self.driver.get_mpo(b.finalize(), iprint=2)

            S2[i] = self.driver.expectation(vec, S2op, vec).real

        return S2
    
    def cal_docc(self, vec):
        nsites = (self.naux + 1) * self.norb

        DOCC = np.zeros(self.norb)
        for i in range(self.norb):
            oplist = _consolidate_static( # to combine reducible op
                number_u(nsites, i) * number_d(nsites, i)
            )
            b = self.driver.expr_builder()
            b2oplist = []
            for op in oplist:
                str, idx, t = op # str could be "+-", "-+", "nn", "+-+-", "++--", "n"
                if self.nspin == 1:
                    spin, idx = map(lambda x: x // nsites, idx), map(lambda x: x % nsites, idx)
                else:
                    spin = [0] * len(idx) # spin: 0 for up (C,D), 1 for down [c,d]
                
                b2str = self.op_map[str][spin]
                b2oplist.append((b2str, idx, t))

                b.add_term(*b2oplist[-1])
            
            DCop = self.driver.get_mpo(b.finalize(), iprint=2)

            DOCC[i] = self.driver.expectation(vec, DCop, vec).real

        return DOCC
    
    @property
    def E(self):
        return self.cal_E(self.ket)
    
    @property
    def S2(self):
        return self.cal_S2(self.ket)
    
    @property
    def RDM(self):
        nsites = (self.naux + 1) * self.norb
        RDM = self.driver.get_1pdm(self.ket)
        if self.nspin == 1:
            RDM = np.kron(0.5 * (RDM[0] + RDM[1]), np.eye(2))
        else:
            # SGF mode, the first
            RDM = RDM.reshape(2, nsites, 2, nsites).transpose(1,0,3,2).reshape(nsites*2, nsites*2)

        if self.decouple_bath:
            RDM = self.recover_decoupled_bath(RDM)

        return RDM
    
    @property
    def docc(self):
        return self.cal_docc(self.ket)
