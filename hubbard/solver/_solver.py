import numpy as np
import copy
from typing import Dict
from hubbard.nao.hf import hartree_fock
from hubbard.nao.tonao import nao_two_chain
from hubbard.operator import Slater_Kanamori, S_z, S_m, S_p, Operator

# base class of all solvers
class Solver(object):
    def __init__(
            self, 
            n_int, 
            n_noint,
            n_elec,
            nspin,
            decouple_bath: bool=False, 
            natural_orbital: bool=False,
            iscomplex=False
            ) -> None:
        
        self.n_int = n_int
        self.n_noint = n_noint
        self.n_elec = n_elec

        self.nspin = nspin
        self.iscomplex = iscomplex
        self.decouple_bath = decouple_bath
        self.natural_orbital = natural_orbital

        self._t = 0.
        self._intparam = {}

    def cal_decoupled_bath(self, T: np.array):
        nsite = self.n_int+self.n_noint
        nint = self.n_int
        dc_T = T.reshape(nsite, 2, nsite, 2).copy()
        if self.nspin == 1:
            assert np.abs(dc_T[:,0,:,0] - dc_T[:,1,:,1]).max() < 1e-7
            _, eigvec = np.linalg.eigh(dc_T[:,0,:,0][nint:,nint:])
            temp_T = dc_T[:,0,:,0]
            temp_T[nint:] = eigvec.conj().T @ temp_T[nint:]
            temp_T[:,nint:] = temp_T[:, nint:] @ eigvec
            dc_T[:,0,:,0] = dc_T[:,1,:,1] = temp_T

            self.transmat = eigvec
        
        elif self.nspin == 2:
            _, eigvecup = np.linalg.eigh(dc_T[:,0,:,0][nint:,nint:])
            _, eigvecdown = np.linalg.eigh(dc_T[:,1,:,1][nint:,nint:])
            temp_T = dc_T[:,0,:,0]
            temp_T[nint:] = eigvecup.conj().T @ temp_T[nint:]
            temp_T[:,nint:] = temp_T[:, nint:] @ eigvecup
            dc_T[:,0,:,0] = temp_T
            temp_T = dc_T[:,1,:,1].copy()
            temp_T[nint:] = eigvecdown.conj().T @ temp_T[nint:]
            temp_T[:,nint:] = temp_T[:, nint:] @ eigvecdown
            dc_T[:,1,:,1] = temp_T

            self.transmat = [eigvecup, eigvecdown]
        
        else:
            dc_T = dc_T.reshape(nsite*2, nsite*2)
            _, eigvec = np.linalg.eigh(dc_T[nint*2:,nint*2:])
            dc_T[nint*2:] = eigvec.conj().T @ dc_T[nint*2:]
            dc_T[:,nint*2:] = dc_T[:,nint*2:] @ eigvec

            self.transmat = eigvec
        
        return dc_T.reshape(nsite*2, nsite*2)
    
    def to_natrual_orbital(self, T: np.array, intparams: Dict[str, float]):
        assert (self.n_noint / self.n_int) - (self.n_noint // self.n_int) < 1e-7, "The non-interacting orbital must be integer number of times of that of the interacting orbitals."

        F,  D, _ = hartree_fock(
            h_mat=T,
            n_imp=self.n_int,
            n_bath=self.n_noint,
            nocc=self.n_elec,
            max_iter=500,
            ntol=self.mutol,
            kBT=self.kBT,
            tol=1e-9,
            **intparams,
            verbose=False,
        )

        D = D.reshape(self.n_int+self.n_noint,2,self.n_int+self.n_noint,2)

        if self.nspin<4:
            assert np.abs(D[:,0,:,1]).max() < 1e-9

        if self.nspin == 1:
            err = np.abs(D[:,0,:,0]-D[:,1,:,1]).max()
            print(D[:,0,:,0]-D[:,1,:,1])
            assert err < 1e-8, "spin symmetry breaking error {}".format(err)

        
        D = D.reshape((self.n_int+self.n_noint)*2, (self.n_int+self.n_noint)*2)

        _, D_meanfield, self.transmat = nao_two_chain(
                                    h_mat=F,
                                    D=D,
                                    n_imp=self.n_int,
                                    n_bath=self.n_noint,
                                    nspin=self.nspin,
                                )
        new_T = self.transmat @ T @ self.transmat.conj().T

        return new_T

    def recover_decoupled_bath(self, T: np.array):
        nsite = self.n_int + self.n_noint
        nint = self.n_int
        rc_T = T.reshape(nsite,2,nsite,2).copy()
        if self.nspin == 1:
            assert np.abs(rc_T[:,0,:,0] - rc_T[:,1,:,1]).max() < 1e-7
            temp_T = rc_T[:,0,:,0]
            temp_T[nint:] = self.transmat @ temp_T[nint:]
            temp_T[:,nint:] = temp_T[:, nint:] @ self.transmat.conj().T
            rc_T[:,0,:,0] = rc_T[:,1,:,1] = temp_T

        if self.nspin == 2:
            temp_T = rc_T[:,0,:,0]
            temp_T[nint:] = self.transmat[0] @ temp_T[nint:]
            temp_T[:,nint:] = temp_T[:, nint:] @ self.transmat[0].conj().T
            rc_T[:,0,:,0] = temp_T
            temp_T = rc_T[:,1,:,1].copy()
            temp_T[nint:] = self.transmat[1] @ temp_T[nint:]
            temp_T[:,nint:] = temp_T[:, nint:] @ self.transmat[1].conj().T
            rc_T[:,1,:,1] = temp_T

        if self.nspin == 4:
            rc_T = rc_T.reshape(nsite*2, nsite*2)
            rc_T[nint*2:] = self.transmat @ rc_T[nint*2:]
            rc_T[:,nint*2:] = rc_T[:,nint*2:] @ self.transmat.conj().T
        
        return rc_T.reshape(nsite*2, nsite*2)

    def _construct_Hop(self, T: np.ndarray, intparam: Dict[str, float]):
        # construct the embedding Hamiltonian
        if self.decouple_bath:
            assert T.shape[0] == (self.n_int + self.n_noint) * 2
            # The first (self.norb, self.norb)*2 is the impurity, which is keeped unchange.
            # We do diagonalization of the other block
            T = self.cal_decoupled_bath(T=T)
        elif self.natural_orbital:
            assert T.shape[0] == (self.n_int + self.n_noint) * 2
            T = self.to_natrual_orbital(T=T, intparams=intparam)

        intparam = copy.deepcopy(intparam)
        intparam["t"] = T.copy()
        self._t = T
        self._intparam = intparam
        nsites = self.n_int + self.n_noint

        int_type = intparam.get("int_type", "Slater_Kanamori")
        if int_type == "Slater_Kanamori":
            _Hop = Slater_Kanamori(
                    nsites=nsites,
                    n_noninteracting=self.n_noint,
                    **intparam
                )
        elif int_type == "QC":
            _Hop = Operator.from_Hqc(h1e=intparam["t"], g2e=intparam["g2e"])
        else:
            raise NotImplementedError

        return _Hop
    
    def get_Hqc(self, symm=False):
        assert hasattr(self, "_Hop"), "The solver need to have the Hamiltonian to get the QC Hamiltonian param."

        return self._Hop.get_Hqc(self.n_int+self.n_noint, symm)


    def get_Hop(self, T, intparam):
        if np.abs(self._t - T).max() > 1e-8 or self._intparam != intparam:
            self._Hop = self._construct_Hop(T, intparam)
            return self._Hop
        else:
            return self._Hop
        
    def solve(self, T, intparam):
        pass

    def get_Eop(self, intonly=False):
        intparam = copy.deepcopy(self._intparam)
        if intonly:
            intparam["t"][self.norb*2:] = 0.
            intparam["t"][:, self.norb*2:] = 0.

        nsites = self.n_int + self.n_noint
        opE = Slater_Kanamori(
                nsites=nsites,
                n_noninteracting=self.n_noint,
                **intparam
            )
        
        return opE

    def get_S2op(self, intonly=False):
        nsites = self.n_int + self.n_noint

        if intonly:
            snorb = self.n_int
        else:
            snorb = self.n_int + self.n_noint

        S_m_ = sum(S_m(nsites, i) for i in range(snorb))
        S_p_ = sum(S_p(nsites, i) for i in range(snorb))
        S_z_ = sum(S_z(nsites, i) for i in range(snorb))

        S2 = S_m_ * S_p_ + S_z_ * S_z_ + S_z_

        return S2