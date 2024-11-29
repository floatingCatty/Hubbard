"""
Input: atomicdata class

output: atomic data class with new node features as R, D
"""
import torch
import torch.nn as nn
from gGA.data import OrbitalMapper
from gGA.nn.ansatz import gGAtomic
from gGA.utils.constants import atomic_num_dict_r
from gGA.data import AtomicDataDict, _keys
from gGA.nn.kinetics import Kinetic
from typing import Union, Dict
from gGA.utils.tools import real_hermitian_basis
from torch.optim import LBFGS, Adam

class GostGutzwiller(object):
    def __init__(
            self, 
            nocc: int, 
            basis: Dict[str, Union[str, list]],
            idx_intorb: Dict[str, list],
            naux: int,
            atomicdata: AtomicDataDict.Type,
            intparams: Dict[str, dict],
            spin_deg: bool=True, 
            device: Union[str, torch.device] = torch.device("cpu"),
            solver: str = "ED",
            kBT: float=0.025,
            delta_deg: float=1e-6,
            decouple_bath: bool=False,
            natural_orbital: bool=False,
            ):
        super(GostGutzwiller, self).__init__()
        # What if all the orbitals are int or none of the orbitals are int?
        
        self.basis = basis
        self.atomicdata = atomicdata
        self.spin_deg = spin_deg
        self.dtype = torch.get_default_dtype()
        self.naux = naux
        self.idx_intorb = idx_intorb
        self.solver = solver
        self.decouple_bath = decouple_bath
        self.natural_orbital = natural_orbital

        if not spin_deg:
            self.idp_phy = self.gGAtomic.idp_phy
        else:
            self.idp_phy = OrbitalMapper(basis=self.gGAtomic.idp_phy.basis, device=device, spin_deg=True)

        if hasattr(self.atomicdata, _keys.ATOMIC_NUMBERS_KEY):
            self.atomic_number = self.atomicdata[AtomicDataDict.ATOMIC_NUMBERS_KEY].flatten().clone()
            self.atomicdata = self.idp_phy(self.atomicdata)
            self.atom_type = self.atomicdata[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        else:
            assert hasattr(self.atomicdata, AtomicDataDict.ATOM_TYPE_KEY), "The atomic number or atom type should be provided."
            self.atom_type = self.atomicdata[AtomicDataDict.ATOM_TYPE_KEY].flatten().clone()
            self.atomic_number = self.idp_phy.untransform_atom(self.atom_type)

        
        # construct Hint_params, [atomwise param, [correalted orbitalwise param, {}]]
        
        self.gGAtomic = gGAtomic(
            basis=basis,
            atomic_number=self.atomic_number,
            idx_intorb=idx_intorb,
            Hint_params=Hint_params,
            naux=naux,
            device=device,
        )

        param = []
        for mulorb in self.gGAtomic.interact_ansatz:
            for singleorb in mulorb.singleOrbs:
                param.append(singleorb.lag_dem_qp)
        self.optimizer = LBFGS(param, lr=1e-1)

        
        """ nocc_phy = n_nonint + n_int
            nocc_aux = n_nonint + n_int_aux
            n_int_aux - n_int = (naux-1)*n_int
            nocc_aux = n_nonint + (naux-1)*n_int + n_int
                     = nocc_phy + (naux-1)*n_int
        """

        n_int = sum([sum(self.nauxorbs[atomic_num_dict_r[an]]) for an in self.atomic_number])

        self.kinetic = Kinetic(
            atomic_number=atomic_number,
            nocc=(naux-1)*n_int+nocc,
            idp=self.idp_aux,
            kBT=kBT,
            device=device,
            idx_intorb=idx_intorb,
            delta_deg=delta_deg,
            )
        
        self.device = device

    @property
    def LAM(self):
        pass

    @property
    def LAM_C(self):
        pass

    @property # beaware that the RDM can be computed from both the qp H and emb H, so we just used the prompt one as the property
    def RDM(self): # but we should keep in mind when convergence is not achieved, RDM would have several VALUE
        pass # TODO: Also, despite this value is stored in gGAtomic part, 
    # it can be updated from qp H, so a update method from qp H to emb H's RDM is needed.

    @property
    def R(self):
        pass

    @property
    def D(self):
        pass

    def update(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        natom = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().shape[0]
        R, LAM = self.gGAtomic.R, self.gGAtomic.LAM
        D, RDM = self.kinetic.update(data, R, LAM)

        # update D, RDM
        self.gGAtomic.update_D(D)
        self.gGAtomic.update_RDM(RDM)

        # update gGA part, for LAM_C, RDM, R
        R, RDM = self.gGAtomic.update(LAM, solver=self.solver, decouple_bath=self.decouple_bath, natural_orbital=self.natural_orbital)
        
        # update LAM
        
        return data

    def fit_LAM(self, niter, data, ref_RDM):
        
        loss_list = []
        def closure():
            self.optimizer.zero_grad()
            LAM = self.gGAtomic.LAM
            R = self.gGAtomic.R
            RDM = self.kinetic.compute_RDM(data, R, LAM)
            for sym in ref_RDM:
                loss += ((ref_RDM[sym] - RDM[sym])**2).sum()

            loss_list.append(loss.item())
            loss.backward()

            return loss
        
        for i in range(niter):
            self.optimizer.step(closure)
            if loss_list[-1] < 1e-6:
                break
    