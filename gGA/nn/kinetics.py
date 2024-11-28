import torch
from gGA.utils import ksum, ksumact, kdsum, kdsumact
from ..model.operators import get_anni
import torch.nn as nn
from gGA.data import OrbitalMapper
from gGA.data import AtomicDataDict
from typing import Tuple, Union, Dict
from gGA.nn.hr2hk import GGAHR2HK
from gGA.utils.tools import float2comlex
import torch.linalg as LA
from gGA.utils.safe_svd import safeSVD
from gGA.utils.constants import atomic_num_dict_r, atomic_num_dict
from gGA.utils.tools import real_hermitian_basis

"""We only need to use single particle basis: |up>, |down>, which is 2 basis for each orbital to solve the problem."""

class Kinetic(object):
    def __init__(
            self,
            nocc,
            atomic_number: torch.Tensor,
            idp_phy: Union[OrbitalMapper, None]=None, 
            basis: Dict[str, Union[str, list]]=None,
            idx_intorb: Dict[str, list]=None,
            spin_deg: bool=False,
            kBT: float = 0.0257,
            device: Union[str, torch.device] = torch.device("cpu"),
            delta_deg=1e-4,
            overlap: bool = False,
            ):
        
        super(Kinetic, self).__init__()
        self.spin_deg = spin_deg
        self.ctype = float2comlex(torch.get_default_dtype())
        self.nocc = nocc
        self.kBT = kBT
        self.atomic_number = atomic_number
        self.idx_intorb = idx_intorb
        
        assert overlap == False, "The overlap is not implemented yet."

        if basis is not None:
            self.idp_phy = OrbitalMapper(basis, method="e3tb", device=self.device, spin_deg=spin_deg)
            if idp_phy is not None:
                assert idp_phy == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp_phy is not None, "Either basis or idp should be provided."
            assert idp_phy.method == "e3tb", "The method of idp should be e3tb."
            self.idp_phy = idp_phy
            if spin_deg is not None:
                assert spin_deg == idp_phy.spin_deg, "The spin_deg of idp and spin_deg should be the same."

        self.hr2hk = GGAHR2HK(
            idp_phy=self.idp_phy,
            overlap=overlap,
            spin_deg=spin_deg,
            device=device
        )

        self.basis = self.idp.basis
        self.delta_deg = delta_deg
        self.device = device

    def update(self, data: AtomicDataDict.Type, R: Dict[str, torch.Tensor], LAM: Dict[str, torch.Tensor]) -> AtomicDataDict.Type:
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested:
            assert kpoints.size(0) == 1
            kpoints = kpoints[0]

        # construct LAM
        H, tkR = self.hr2hk(data, R, LAM)

        nk = kpoints.size(0)
        eigval, eigvec = torch.linalg.eigh(H)
        norb = eigval.shape[1]
        eigvec = eigvec.reshape(nk, -1, norb).transpose(1,2)
        
        E_fermi = torch.sort(eigval.flatten())[0][self.nocc*nk-1:self.nocc*nk+1].sum() * 0.5
        # factor = fermi_dirac(eigval, E_fermi, self.kBT).detach()
        mask = eigval < E_fermi
        nstates = mask.sum()
        if nstates < nk * self.nocc:
            state_left = nk * self.nocc - nstates
            mask_left = eigval.ge(E_fermi) * eigval.lt(E_fermi+self.delta_deg)
            ndegstates = mask_left.sum()
        else:
            mask_left = None

        vec_m = eigvec[mask]
        real_C2 = torch.einsum("ni, nj->nij", vec_m.conj(), vec_m).sum(dim=0) # nstates, norb*2, norb*2
        if mask_left is not None:
            vec_ml = eigvec[mask_left]
            real_C2 += (state_left/ndegstates) * torch.einsum("ni, nj->nij", vec_ml.conj(), vec_ml).sum(dim=0)
            T = (eigval[mask].sum() + (state_left/ndegstates) * eigval[mask_left].sum()) / nk
        else:
            T = eigval[mask].sum() / nk
        real_C2 = real_C2.real / nk

        data[AtomicDataDict.REDUCED_DENSITY_MATRIX_KEY] = real_C2

        # compute D and decompose D


        return data
        
    


def fermi_dirac(energy, E_fermi, kBT):
    return 1/(torch.exp((energy-E_fermi)/kBT) + 1)

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

if __name__ == "__main__":
    kin = Kinetics(
        t=torch.eye(4).reshape(1,1,2,1,2),
        R=torch.tensor([[0,0,0]]),
        kpoints=torch.tensor([[0,0,0]]),
        kspace=True
    )

    H = kin.get_hamiltonian()

    print(H)
