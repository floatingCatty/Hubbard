import numpy as np
import copy
from typing import Dict
from gGA.operator import Slater_Kanamori, create_d, annihilate_d, create_u, annihilate_u, number_d, number_u

class ED_solver(object):
    def __init__(
            self, 
            norb, 
            naux, 
            nspin, 
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
        
        if self.nspin == 1:
            # Nparticle = [(self.norb*(self.naux+1)-i,i) for i in range(0,self.norb*(self.naux+1)+1)]
            self.Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]
        else:
            # self.Nparticle = [(self.norb*(self.naux+1)//2,self.norb*(self.naux+1)//2)]
            self.Nparticle = [(self.norb*(self.naux+1)-i,i) for i in range(0,self.norb*(self.naux+1)+1)]

        self._t = 0.
        self._intparam = {}

    def _construct_Hemb(self, T: np.ndarray, intparam: Dict[str, float]):
        # construct the embedding Hamiltonian

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
            RDM = self.cal_RDM(vec=vec)

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
    
    @property
    def E(self):
        return self.cal_E(self.vec)
    
    @property
    def RDM(self):
        return self.cal_RDM(self.vec)
    
    @property
    def docc(self):
        return self.cal_docc(self.vec)
