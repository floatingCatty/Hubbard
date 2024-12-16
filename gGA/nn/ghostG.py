"""
Input: atomicdata class

output: atomic data class with new node features as R, D
"""
import torch
from gGA.nn.ansatz import gGAtomic
from gGA.utils.constants import atomic_num_dict_r
from gGA.data import AtomicDataDict, _keys
from gGA.nn.kinetics import Kinetic
from typing import Union, Dict

class GhostGutzwiller(object):
    def __init__(
            self, 
            atomic_number: torch.Tensor,
            nocc: Dict[str, list], # ep. {"Si": [2, 2, 0]}
            basis: Dict[str, Union[str, list]], # ep. {"Si": ["s", "p", "d"]}
            idx_intorb: Dict[str, list],
            naux: int,
            intparams: Dict[str, dict],
            nspin: int=1, 
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
        self.nspin = nspin
        self.dtype = torch.get_default_dtype()
        self.naux = naux
        self.nocc = nocc
        self.idx_intorb = idx_intorb
        self.solver = solver
        self.decouple_bath = decouple_bath
        self.natural_orbital = natural_orbital
        self.intparams = intparams
        self.atomic_number = atomic_number
        self.overlap = overlap
        self.device = device
        
        self.gGAtomic = gGAtomic(
            basis=basis,
            atomic_number=self.atomic_number,
            idx_intorb=idx_intorb,
            intparams=self.intparams,
            naux=naux,
            nocc=nocc,
            device=device,
            nspin=nspin,
        )

        # do nocc counting
        nele_phy = 0
        nele_intphy = 0
        nele_intaux = 0
        for i in atomic_number.tolist():
            sym = atomic_num_dict_r[i]
            nele_phy += sum(nocc[sym])
            nele_intphy += sum([nocc[sym][idx] for idx in self.idx_intorb[sym]])
            nele_intaux += sum([(naux-1) * self.gGAtomic.idp_phy.listnorbs[sym][idx] + nocc[sym][idx] for idx in self.idx_intorb[atomic_num_dict_r[i]]])
        
        """ nocc_phy = nocc_phynonint + nocc_phyint
            nocc_aux = nocc_phynonint + nocc_int_aux
            nocc_int_aux - nocc_phyint = (naux-1)*norb_phyint
            nocc_int_aux = nocc_phyint + (naux-1)*norb_phyint
                         = nocc_nonint + (naux-1)*n_int + n_int
                     = nocc_phy + (naux-1)*n_int
        """

        self.kinetic = Kinetic(
            nocc=nele_phy - nele_intphy + nele_intaux,
            naux=naux,
            basis=basis,
            idx_intorb=idx_intorb,
            nspin=nspin,
            kBT=kBT,
            soc=False,
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
            # self.gGAtomic.fix_gauge()

        R, LAM = self.gGAtomic.R, self.gGAtomic.LAM
        phy_onsite, D, RDM, E_fermi = self.kinetic.update(data, R, LAM)
        # update D, RDM
        self.gGAtomic.update_D(D)
        self.gGAtomic.update_RDM(RDM)
        # self.gGAtomic.update_LAM(LAM)

        # update gGA part, for LAM_C, RDM, R
        R_new, LAM_new = self.gGAtomic.update(phy_onsite, E_fermi=0., solver=self.solver, decouple_bath=self.decouple_bath, natural_orbital=self.natural_orbital)
        
        RDM_emb = self.gGAtomic.RDM
        print("DM_emb: ", torch.linalg.eigvalsh(RDM_emb["C"][0]))

        # compute error
        err = 0

        with torch.no_grad():
            for sym in R:
                err = max(err, (R_new[sym] - R[sym]).abs().max().item())
        
        return err, RDM_emb