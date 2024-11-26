import torch
from gGA.utils import ksum, ksumact, kdsum, kdsumact
from ..model.operators import get_anni
import torch.nn as nn
from gGA.data import OrbitalMapper
from gGA.data import AtomicDataDict
from typing import Tuple, Union, Dict
from gGA.utils.tools import float2comlex
import torch.linalg as LA
from gGA.utils.tools import real_hermitian_basis

"""We only need to use single particle basis: |up>, |down>, which is 2 basis for each orbital to solve the problem."""

class Kinetic(nn.Module):
    def __init__(
            self, 
            atomic_number: torch.Tensor, 
            nocc, 
            idp: Union[OrbitalMapper, None]=None, 
            basis: Dict[str, Union[str, list]]=None,
            spin_deg: bool=False,
            kBT: float = 0.0257,
            device: Union[str, torch.device] = torch.device("cpu"),
            delta_deg=1e-4,
            ):
        
        super(Kinetic, self).__init__()
        self.spin_deg = spin_deg
        self.ctype = float2comlex(torch.get_default_dtype())
        self.nocc = nocc
        self.kBT = kBT
        self.atomic_number = atomic_number
        
        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device, spin_deg=spin_deg)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            assert idp.method == "e3tb", "The method of idp should be e3tb."
            self.idp = idp
            if spin_deg is not None:
                assert spin_deg == idp.spin_deg, "The spin_deg of idp and spin_deg should be the same."

        self.totalorb = self.idp.atom_norb[self.idp.transform(atomic_number).flatten()].sum()
        self.pvec = 2*min(nocc, self.totalorb)

        self.basis = self.idp.basis
        self.totalorb = self.totalorb
        self.lag_den_emb
        self.delta_deg = delta_deg
        # self.network = nn.Sequential(
        #     nn.Linear(3, neurons[0], dtype=self.ctype),
        #     nn.Tanh(),
        #     nn.Linear(neurons[0], neurons[1], dtype=self.ctype),
        #     nn.Tanh(),
        #     nn.Linear(neurons[1], neurons[2], dtype=self.ctype),
        #     nn.Tanh(),
        #     nn.Linear(neurons[2], self.totalorb*2*self.pvec, dtype=self.ctype)
        # )
        # self.network.to(device)
        # self.nk = nk
        self.device = device
        # self.W = torch.nn.Parameter(torch.randn(nk, self.totalorb*2, self.pvec, dtype=self.ctype, device=self.device))
        
       

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested:
            assert kpoints.size(0) == 1
            kpoints = kpoints[0]
        
        # # W = self.network(kpoints.to(self.ctype)).reshape(-1, self.totalorb*2, self.pvec) # [nk, norb*2, pvec]
        
        
        pesudo_eigenvec = LA.qr(self.W).Q.transpose(1,2) # the wave vector [nkpoints, pvec, 2*norb]
        nk = kpoints.size(0)

        # compute kinetical energy
        eigval = torch.bmm(pesudo_eigenvec.conj(), data[AtomicDataDict.HAMILTONIAN_KEY]) # [nk, pvec, norb*2] x [nk, norb*2, norb*2] = [nk, pvec, norb*2]
        eigval = (eigval * pesudo_eigenvec).sum(dim=-1).real # [nk, pvec]

        E_fermi = torch.sort(eigval.flatten())[0][self.nocc*nk-1:self.nocc*nk+1].sum() * 0.5
        # factor = fermi_dirac(eigval, E_fermi, self.kBT).detach()
        mask = eigval < E_fermi

        T = (eigval[mask]).sum().real / nk
        
        reduced_density_matrix = pesudo_eigenvec[mask]
        reduced_density_matrix = torch.einsum("ni, nj->nij", reduced_density_matrix.conj(), reduced_density_matrix).real.sum(dim=0) / nk # [norb*2, norb*2]

        data[AtomicDataDict.TOTAL_ENERGY_KEY] = T
        data[AtomicDataDict.REDUCED_DENSITY_MATRIX_KEY] = reduced_density_matrix
        
        # nk = kpoints.size(0)
        # eigval, eigvec = torch.linalg.eigh(data[AtomicDataDict.HAMILTONIAN_KEY])
        # eigval = eigval.flatten()
        # eigvec = eigvec.reshape(nk*2*self.totalorb, -1)
        
        # E_fermi = torch.sort(eigval.flatten())[0][self.nocc*nk-1:self.nocc*nk+1].sum() * 0.5
        # # factor = fermi_dirac(eigval, E_fermi, self.kBT).detach()
        # mask = eigval < E_fermi
        # nstates = mask.sum()
        # if nstates < nk * self.nocc:
        #     state_left = nk * self.nocc - nstates
        #     mask_left = eigval.ge(E_fermi) * eigval.lt(E_fermi+self.delta_deg)
        #     ndegstates = mask_left.sum()
        # else:
        #     mask_left = None

        # vec_m = eigvec[mask]
        # real_C2 = torch.einsum("ni, nj->nij", vec_m.conj(), vec_m).sum(dim=0) # nstates, norb*2, norb*2
        # if mask_left is not None:
        #     vec_ml = eigvec[mask_left]
        #     real_C2 += (state_left/ndegstates) * torch.einsum("ni, nj->nij", vec_ml.conj(), vec_ml).sum(dim=0)
        #     T = (eigval[mask].sum() + (state_left/ndegstates) * eigval[mask_left].sum()) / nk
        # else:
        #     T = eigval[mask].sum() / nk
        # real_C2 = real_C2.real / nk

        # data[AtomicDataDict.TOTAL_ENERGY_KEY] = T
        # data[AtomicDataDict.REDUCED_DENSITY_MATRIX_KEY] = real_C2

        # pesudo_eigenvec = pesudo_eigenvec.reshape(nk*self.pvec, -1)
        # reduced_density_matrix = pesudo_eigenvec * factor.unsqueeze(1)

        return data
        
    


def fermi_dirac(energy, E_fermi, kBT):
    return 1/(torch.exp((energy-E_fermi)/kBT) + 1)

if __name__ == "__main__":
    kin = Kinetics(
        t=torch.eye(4).reshape(1,1,2,1,2),
        R=torch.tensor([[0,0,0]]),
        kpoints=torch.tensor([[0,0,0]]),
        kspace=True
    )

    H = kin.get_hamiltonian()

    print(H)
