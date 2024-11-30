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
from torch.optim import LBFGS, Adam, RMSprop

class GhostGutzwiller(object):
    def __init__(
            self, 
            atomic_number: torch.Tensor,
            nocc: int,
            basis: Dict[str, Union[str, list]],
            idx_intorb: Dict[str, list],
            naux: int,
            intparams: Dict[str, dict],
            spin_deg: bool=True, 
            device: Union[str, torch.device] = torch.device("cpu"),
            solver: str = "ED",
            kBT: float=0.025,
            delta_deg: float=1e-6,
            overlap: bool=False,
            decouple_bath: bool=False,
            natural_orbital: bool=False,
            ):
        super(GhostGutzwiller, self).__init__()
        # What if all the orbitals are int or none of the orbitals are int?
        
        self.basis = basis
        self.spin_deg = spin_deg
        self.dtype = torch.get_default_dtype()
        self.naux = naux
        self.idx_intorb = idx_intorb
        self.solver = solver
        self.decouple_bath = decouple_bath
        self.natural_orbital = natural_orbital
        self.intparams = intparams
        self.atomic_number = atomic_number
        self.overlap = overlap
        self.device = device

        # construct Hint_params, [atomwise param, [correalted orbitalwise param, {}]]
        
        self.gGAtomic = gGAtomic(
            basis=basis,
            atomic_number=self.atomic_number,
            idx_intorb=idx_intorb,
            intparams=self.intparams,
            naux=naux,
            device=device,
            spin_deg=spin_deg,
        )

        param = []
        for mulorb in self.gGAtomic.interact_ansatz:
            for singleorb in mulorb.singleOrbs:
                singleorb.lag_den_qp = nn.Parameter(singleorb.lag_den_qp)
                param.append(singleorb.lag_den_qp)
        # self.optimizer = LBFGS(param, max_iter=10)
        self.optimizer = Adam(param, lr=1e-1)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=100, verbose=True)
        
        """ nocc_phy = n_nonint + n_int
            nocc_aux = n_nonint + n_int_aux
            n_int_aux - n_int = (naux-1)*n_int
            nocc_aux = n_nonint + (naux-1)*n_int + n_int
                     = nocc_phy + (naux-1)*n_int
        """

        n_int = sum([sum(self.gGAtomic.idp_phy.listnorbs[atomic_num_dict_r[int(an)]]) for an in self.atomic_number])

        self.kinetic = Kinetic(
            nocc=(naux-1)*n_int+nocc,
            basis=basis,
            idx_intorb=idx_intorb,
            spin_deg=spin_deg,
            kBT=kBT,
            device=device,
            overlap=overlap,
            delta_deg=delta_deg,
            )
        
        
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
        with torch.no_grad():
            R, LAM = self.gGAtomic.R, self.gGAtomic.LAM
            phy_onsite, D, RDM = self.kinetic.update(data, R, LAM)

            # update D, RDM
            self.gGAtomic.update_D(D)
            self.gGAtomic.update_RDM(RDM)

            # update gGA part, for LAM_C, RDM, R
            R, RDM = self.gGAtomic.update(phy_onsite, LAM, solver=self.solver, decouple_bath=self.decouple_bath, natural_orbital=self.natural_orbital)

        # update LAM
        self.fit_LAM(10000, data, RDM)
        
        return data

    def fit_LAM(self, niter, data, ref_RDM):
        
        # loss_list = []
        # def closure():
        #     loss = 0.
        #     self.optimizer.zero_grad()
        #     LAM = self.gGAtomic.LAM
        #     R = self.gGAtomic.R
        #     RDM = self.kinetic.compute_RDM(data, R, LAM)
        #     for sym in ref_RDM:
        #         loss += ((ref_RDM[sym] - RDM[sym])**2).sum()

        #     loss_list.append(loss.item())
        #     loss.backward()

        #     return loss
        
        # for i in range(niter):
        #     self.optimizer.step(closure)
        #     if loss_list[-1] < 1e-6:
        #         break
        #     else:
        #         print(loss_list[-1])

        for _ in range(niter):
            self.optimizer.zero_grad()
            LAM = self.gGAtomic.LAM
            R = self.gGAtomic.R
            RDM = self.kinetic.compute_RDM(data, R, LAM)
            loss = 0.
            max_div = 0.
            for sym in ref_RDM:
                loss += ((ref_RDM[sym] - RDM[sym])**2).sum()
                max_div = max(max_div, (ref_RDM[sym] - RDM[sym]).abs().max().item())

            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step(loss.item())

            if loss.item() < 1e-6:
                break
            else:
                print(max_div, self.lr_scheduler._last_lr)
    