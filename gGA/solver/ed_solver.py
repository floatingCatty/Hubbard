import numpy as np
import copy
from typing import Dict
from gGA.operator import Slater_Kanamori, create_d, annihilate_d, create_u, annihilate_u, number_d, number_u, S_z, S_m, S_p
from gGA.nao.hf import hartree_fock
from gGA.nao.tonao import nao_two_chain

class ED_solver(object):
    def __init__(
            self, 
            norb, 
            naux, 
            nspin, 
            kBT: float=0.025,
            mutol: float=1e-4,
            decouple_bath: bool=False, 
            natural_orbital: bool=False, 
            iscomplex=False
            ) -> None:
        
        self.norb = norb
        self.naux = naux
        self.nspin = nspin
        self.decouple_bath = decouple_bath
        self.natural_orbital = natural_orbital
        self.iscomplex = iscomplex
        self.mutol = mutol
        self.kBT = kBT
        
        if self.nspin == 1:
            # Nparticle = [(self.norb*(self.naux+1)-i,i) for i in range(0,self.norb*(self.naux+1)+1)]
            self.Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]
        else:
            # self.Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]
            self.Nparticle = [(self.norb*(self.naux+1)-i,i) for i in range(0,self.norb*(self.naux+1)+1)]

        self._t = 0.
        self._intparam = {}

    def to_natrual_orbital(self, T: np.array, intparams: Dict[str, float]):
        F,  D, _ = hartree_fock(
            h_mat=T,
            n_imp=self.norb,
            n_bath=self.naux*self.norb,
            nocc=(self.naux+1)*self.norb, # always half filled
            max_iter=500,
            ntol=self.mutol,
            kBT=self.kBT,
            tol=1e-6,
            **intparams,
            verbose=False,
        )

        D = D.reshape((self.naux+1)*self.norb,2,(self.naux+1)*self.norb,2)

        if self.nspin<4:
            assert np.abs(D[:,0,:,1]).max() < 1e-9

        if self.nspin == 1:
            err = np.abs(D[:,0,:,0]-D[:,1,:,1]).max()
            assert err < 1e-9, "spin symmetry breaking error {}".format(err)
        
        D = D.reshape((self.naux+1)*self.norb*2, (self.naux+1)*self.norb*2)

        _, D_meanfield, self.transmat = nao_two_chain(
                                    h_mat=F,
                                    D=D,
                                    n_imp=self.norb,
                                    n_bath=self.naux*self.norb,
                                    nspin=self.nspin,
                                )
        new_T = self.transmat @ T @ self.transmat.conj().T

        return new_T

    def _construct_Hemb(self, T: np.ndarray, intparam: Dict[str, float]):
        # construct the embedding Hamiltonian
        if self.natural_orbital:
            assert T.shape[0] == (self.naux+1) * self.norb * 2
            T = self.to_natrual_orbital(T=T, intparams=intparam)

        intparam = copy.deepcopy(intparam)
        intparam["t"] = T.copy()
        self._t = T
        self._intparam = intparam

        self._Hemb = Slater_Kanamori(
            nsites=self.norb*(self.naux+1),
            n_noninteracting=self.norb*self.naux,
            **intparam
        ) 
        # we should notice that the spinorbital are not adjacent in the quspin hamiltonian, 
        # so properties computed from this need to be transformed.

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

        val, vec = Hemb.diagonalize(
            nsites=nsites, 
            Nparticle=self.Nparticle,
            iscomplex=self.iscomplex,
        )

        self.vec = vec

        if return_RDM:
            # import matplotlib.pyplot as plt
            RDM = self.cal_RDM(vec=vec)

            # plt.matshow(RDM, cmap="bwr", vmax=0.1, vmin=-0.1)
            # plt.show()
            if self.natural_orbital:
                RDM = self.transmat.conj().T @ RDM @ self.transmat

        return RDM
    
    def cal_RDM(self, vec):
        vec = np.asarray(vec)
        nsites = self.norb*(self.naux+1)

        # # compute RDM
        RDM = np.zeros(((self.naux+1)*self.norb, 2, (self.naux+1)*self.norb, 2))
        if self.iscomplex:
            RDM = RDM + 0j
        for a in range((self.naux+1) * self.norb):
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
                        
                        v = op.get_quspin_op(nsites, self.Nparticle).expt_value(vec)
                        RDM[a, s, b, s_] = v
                        RDM[b, s_, a, s] = v.conj()

        return RDM.reshape(2*nsites, 2*nsites)
    
    def cal_E(self, vec):
        intparam = copy.deepcopy(self._intparam)
        intparam["t"][self.norb*2:] = 0.
        intparam["t"][:, self.norb*2:] = 0.

        Hemb = Slater_Kanamori(
            nsites=self.norb*(self.naux+1),
            n_noninteracting=self.norb*self.naux,
            **intparam
        )

        E = Hemb.get_quspin_op(self.norb*(self.naux+1), self.Nparticle).expt_value(vec)

        return E
    
    def cal_docc(self, vec):
        vec = np.asarray(vec)
        nsites = self.norb*(self.naux+1)

        nocc = np.zeros(self.norb)
        
        for i in range(self.norb):
            op = number_u(nsites, i) * number_d(nsites, i)

            v = op.get_quspin_op(nsites, self.Nparticle).expt_value(vec)
            nocc[i] = v.real
        
        return nocc

    def cal_S2(self, vec):
        vec = np.asarray(vec)
        nsites = self.norb*(self.naux+1)

        S2 = np.zeros(self.norb)
        for i in range(self.norb):
            op = S_m(nsites, i) * S_p(nsites, i) + S_z(nsites, i) * S_z(nsites, i) + S_z(nsites, i)
            v = op.get_quspin_op(nsites, self.Nparticle).expt_value(vec)
            S2[i] = v.real
        
        return S2
    
    @property
    def E(self):
        return self.cal_E(self.vec)

    @property
    def S2(self):
        return self.cal_S2(self.vec)
    
    @property
    def RDM(self):
        return self.cal_RDM(self.vec)
    
    @property
    def docc(self):
        return self.cal_docc(self.vec)
