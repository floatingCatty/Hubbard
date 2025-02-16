"""
Input: atomicdata class

output: atomic data class with new node features as R, D
"""
import numpy as np
from gGA.gutz.ansatz import gGAtomic
from gGA.utils.constants import atomic_num_dict_r, atomic_num_dict
from gGA.data import AtomicDataDict, _keys
from gGA.gutz.kinetics import Kinetic
from typing import Union, Dict
from copy import deepcopy
import os
from scipy.linalg import block_diag

class GhostGutzwiller(object):
    def __init__(
            self, 
            atomic_number: np.ndarray,
            nocc: int, # ep. {"Si": [2, 2, 0]}
            basis: Dict[str, Union[str, list]], # ep. {"Si": ["s", "p", "d"]}
            idx_intorb: Dict[str, list],
            naux: int,
            intparams: Dict[str, dict],
            nspin: int=1,
            solver: str = "ED",
            kBT: float=0.025,
            delta_deg: float=1e-6,
            mutol: float=1e-4,
            overlap: bool=False,
            decouple_bath: bool=False,
            natural_orbital: bool=False,
            solver_options: dict={},
            mixer_options: dict={},
            iscomplex: bool=False,
            dtype=np.float64
            ):
        super(GhostGutzwiller, self).__init__()
        # What if all the orbitals are int or none of the orbitals are int?

        self.state = {
            "atomic_number": atomic_number,
            "nocc": nocc,
            "basis": basis,
            "idx_intorb": idx_intorb,
            "naux": naux,
            "intparams": intparams,
            "nspin": nspin,
            "solver": solver,
            "kBT": kBT,
            "delta_deg": delta_deg,
            "mutol": mutol,
            "overlap": overlap,
            "decouple_bath": decouple_bath,
            "natural_orbital": natural_orbital,
            "solver_options": solver_options,
            "mixer_options": mixer_options,
            "iscomplex": iscomplex,
            "dtype": dtype
        }
        
        self.basis = basis
        self.nspin = nspin
        self.spin_deg = nspin <= 1
        self.dtype = dtype
        self.naux = naux
        self.nocc = nocc
        self.idx_intorb = idx_intorb
        self.solver = solver
        self.decouple_bath = decouple_bath
        self.natural_orbital = natural_orbital
        self.intparams = intparams
        self.atomic_number = atomic_number
        self.overlap = overlap
        self.solver_options = solver_options
        self.mixer_options = mixer_options
        self.iscomplex = iscomplex
        
        self.gGAtomic = gGAtomic(
            basis=basis,
            atomic_number=self.atomic_number,
            idx_intorb=idx_intorb,
            solver=solver,
            naux=naux,
            nspin=nspin,
            dtype=self.dtype,
            kBT=kBT,
            mutol=mutol,
            decouple_bath=self.decouple_bath,
            natural_orbital=self.natural_orbital,
            mixer_options=self.mixer_options,
            solver_options=self.solver_options,
            iscomplex=iscomplex,
        )

        # do nocc counting
        nint = 0
        for i in atomic_number.tolist():
            sym = atomic_num_dict_r[i]
            if self.idx_intorb.get(sym) is not None:
                nint += sum([self.gGAtomic.idp_phy.listnorbs[sym][idx] for idx in self.idx_intorb[sym]])
        
        """ nocc_phy = nocc_phynonint + nocc_phyint
            nocc_aux = nocc_phynonint + nocc_int_aux
            nocc_int_aux - nocc_phyint = (naux-1)*norb_phyint
            nocc_int_aux = nocc_phyint + (naux-1)*norb_phyint
                         = nocc_nonint + (naux-1)*n_int + n_int
                     = nocc_phy + (naux-1)*n_int
        """

        self.kinetic = Kinetic(
            nocc=nocc + (self.naux-1) * nint,
            naux=naux,
            basis=basis,
            idx_intorb=idx_intorb,
            nspin=nspin,
            kBT=kBT,
            mutol=mutol,
            soc=False,
            dtype=self.dtype,
            overlap=overlap,
            delta_deg=delta_deg,
            iscomplex=iscomplex,
            )
        
        self.nauxorbs = self.kinetic.nauxorbs
        self.idp_phy = self.kinetic.idp_phy
        # build a map to map non-interacting aux RDM to phy RDM
        # Here the RDM should have both spin up and down, so we need to carefully repeat the map when spin_deg is True
        self.map_rdm_aux = {}
        self.map_rdm_phy = {}
        
        for sym in self.idp_phy.type_names:
            self.map_rdm_aux[sym] = np.ones(sum(self.nauxorbs[sym])*2, dtype=np.bool)
            self.map_rdm_phy[sym] = np.ones(sum(self.idp_phy.listnorbs[sym])*2, dtype=np.bool)
            if self.idx_intorb.get(sym) is not None:
                for io, orb in enumerate(self.idx_intorb[sym]):
                    norb = self.nauxorbs[sym][orb]
                    snorb = sum(self.nauxorbs[sym][:orb])

                    self.map_rdm_aux[sym][snorb*2:snorb*2+norb*2] = False

                    norb = self.idp_phy.listnorbs[sym][orb]
                    snorb = sum(self.idp_phy.listnorbs[sym][:orb])
                    self.map_rdm_phy[sym][snorb*2:snorb*2+norb*2] = False

    @property
    def LAM(self):
        return self.gGAtomic.LAM

    @property
    def LAM_C(self):
        return self.gGAtomic.LAM_C

    @property # beaware that the RDM can be computed from both the qp H and emb H, so we just used the prompt one as the property
    def RDM(self): # but we should keep in mind when convergence is not achieved, RDM would have several VALUE
        RDM = deepcopy(self.RDM_kin) # start from the RDM computed from kinetical part
        for sym in RDM:
            new_rdm = np.zeros((RDM[sym].shape[0], sum(self.gGAtomic.idp_phy.listnorbs[sym])*2,sum(self.gGAtomic.idp_phy.listnorbs[sym])*2), dtype=RDM[sym].dtype)
            map_nonint = self.map_rdm_phy[sym]
            map_nonint_aux = self.map_rdm_aux[sym]
            new_rdm[:,map_nonint[:, None] * map_nonint[None, :]] = RDM[sym][:,map_nonint_aux[:,None]*map_nonint_aux[None,:]]
            RDM[sym] = new_rdm
        
        for sym in self.idx_intorb.keys():
            for ita, ia in enumerate(np.arange(len(self.atomic_number))[self.atomic_number == atomic_num_dict[sym]]):
                for i, into in enumerate(self.idx_intorb[sym]):
                    norb = self.gGAtomic.idp_phy.listnorbs[sym][into]
                    snorb = sum(self.gGAtomic.idp_phy.listnorbs[sym][:into])
                    # RDM[sym][ita][snorb*2:snorb*2+norb*2,:] = 0.
                    # RDM[sym][ita][:,snorb*2:snorb*2+norb*2] = 0.
                    RDM[sym][ita][snorb*2:snorb*2+norb*2,snorb*2:snorb*2+norb*2] = self.gGAtomic.interact_ansatz[ia].fRDM[i][:norb*2, :norb*2] # might be wrong

        return RDM
    
    # TODO: Also, despite this value is stored in gGAtomic part, 
    # it can be updated from qp H, so a update method from qp H to emb H's RDM is needed.

    @property
    def R(self):
        return self.gGAtomic.R

    @property
    def D(self):
        return self.gGAtomic.D

    def update(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # self.gGAtomic.fix_gauge()
        self.data = data
        R, LAM = self.gGAtomic.R, self.gGAtomic.LAM
        phy_onsite, D, self.RDM_kin, self.E_fermi = self.kinetic.update(data, R, LAM)
        # update D, RDM
        self.gGAtomic.update_D(D)
        self.gGAtomic.update_RDM(self.RDM_kin)
        # self.gGAtomic.update_LAM(LAM)

        # update gGA part, for LAM_C, RDM, R
        self.gGAtomic.update(phy_onsite, self.intparams, E_fermi=0.)
        
        RDM_emb = self.gGAtomic.RDM

        # compute error
        err = 0
        R_new = self.gGAtomic.R
        LAM_new = self.gGAtomic.LAM

        for sym in R:
            err = max(err, np.abs(R_new[sym] - R[sym]).max())
            err = max(err, np.abs(LAM_new[sym] - LAM[sym]).max())
    
        return err, RDM_emb
    
    def run(self, data, maxiter, tol):
        for i in range(maxiter):
            err, RDM = self.update(data)

            if err < tol:
                print("Convergence achieved!\n")
                break
            else:
                print(" -- Current error: {:.5f}".format(err))
        
        print("Convergened Density: ", self.RDM)
        
        return RDM

    def update_intparam(self, intparam):
        self.intparams = deepcopy(intparam)

        return True

    def reset(self):
        self.gGAtomic.reset()
        return True
    
    def save(self, f, prefix=None):
        states = {
            "R": self.gGAtomic.R,
            "D": self.gGAtomic.D,
            "LAM": self.gGAtomic.LAM,
            "LAM_C": self.gGAtomic.LAM_C,
            "RDM": self.gGAtomic.RDM,
            "RDM_kin": self.RDM_kin,
            "E_fermi": self.E_fermi,
            "intparams": self.intparams,
            "state": self.state,
            "data": self.data
            }
        
        na = "state"
        if isinstance(prefix, str):
            na = na + prefix + ".npz"
        else:
            na = na + ".npz"

        np.savez(os.path.join(f, na), **states)

        return True
    
    def load(self, f):
        obj = np.load(f, allow_pickle=True)
        self.gGAtomic.update_RDM(obj["RDM"].item())
        self.gGAtomic.update_LAM(obj["LAM"].item())
        self.gGAtomic.update_LAM_C(obj["LAM_C"].item())
        self.gGAtomic.update_D(obj["D"].item())
        self.gGAtomic.update_R(obj["R"].item())
        # self.intparams = obj["intparams"]

        self.RDM_kin = obj["RDM_kin"].item()
        self.E_fermi = obj["E_fermi"].item()

        return True
    
    @classmethod
    def from_ckpt(cls, f):
        obj = np.load(f, allow_pickle=True)
        state = obj["state"].item()
        ga = cls(**state)
        data = ga.load(f)
        
        return ga, data
    
    def compute_GF(self, Es, data: AtomicDataDict.Type, eta=1e-5):
        R = self.gGAtomic.R
        _, H, _, S = self.kinetic._compute_H(data=data, R=R, LAM=self.gGAtomic.LAM)
        n = H.shape[1]
        Es = Es.reshape(-1)

        Rs = []
        type_count = [0] * len(self.kinetic.idp_phy.type_names)
        for i, at in enumerate(data[AtomicDataDict.ATOM_TYPE_KEY].flatten()):
            idx = type_count[at]
            sym = self.kinetic.idp_phy.type_names[at]
            if sym in self.idx_intorb.keys():
                Rs.append(R[sym][idx].conj().T)
            else:
                Rs.append(np.eye(sum(self.nauxorbs[sym])*2))

            type_count[at] += 1
        
        Rs = block_diag(*Rs).astype(H.dtype)
        RsH = Rs.conj().T

        if S is None:
            GF = Rs[None,None,...] @ np.linalg.inv((Es+1j*eta)[None,:,None,None] * np.eye(n).astype(H.dtype)[None,None,:,:] - H[:,None,...]) @ RsH[None,None,...]
        else:
            GF = Rs[None,None,...] @ np.linalg.inv((Es+1j*eta)[None,:,None,None] * S - H[:,None,...]) @ RsH[None,None,...]

        return GF
