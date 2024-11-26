"""
Input: atomicdata class

output: atomic data class with new node features as R, D
"""
import torch
import torch.nn as nn
from gGA.data import OrbitalMapper
from gGA.nn.ansatz import gGAtomic
from gGA.utils.constants import atomic_num_dict_r
from gGA.data import AtomicDataDict
from gGA.nn.hr2hk import GGAHR2HK
from gGA.nn.kinetics import Kinetic
from typing import Union, Dict
from gGA.utils.tools import real_hermitian_basis

class GostGutzwiller(nn.Module):
    def __init__(
            self, 
            atomic_number, 
            nocc, 
            basis,
            idx_intorb, 
            naux,
            Hint_params, 
            spin_deg=True, 
            device: Union[str, torch.device] = torch.device("cpu"),
            kBT=0.025,
            ):
        super(GostGutzwiller, self).__init__()
        # What if all the orbitals are int or none of the orbitals are int?
        
        self.basis = basis
        self.atomic_number = atomic_number
        self.spin_deg = spin_deg
        self.dtype = torch.get_default_dtype()
        self.naux = naux
        self.idx_intorb = idx_intorb
        self.gGAtomic = gGAtomic(
            basis=basis,
            atomic_number=atomic_number,
            idx_intorb=idx_intorb,
            Hint_params=Hint_params,
            naux=naux,
            device=device,
        )

        self.idp_phy = self.gGAtomic.idp_phy
        self.idp_aux = OrbitalMapper(basis=self.gGAtomic.nauxorbs, device=device, spin_deg=False)
        """ build a basis map that can extract the auxilary interaction basis from the full auxilary basis would help to address the order problem
        since aux int orbital in idp_intaux and idp_aux must have the same ascend order 
        The map can be build in full basis level, instead of the atomic specific basis, to map them in one shot.
        """

        if spin_deg == True:
            self.hr2hk = GGAHR2HK(
                idp_phy=OrbitalMapper(basis=self.idp_phy.basis, device=device, spin_deg=True),
                idp_aux=self.idp_aux,
                naux=naux,
                device=device,
            )
        else:
            self.hr2hk = GGAHR2HK(
                idp_phy=self.idp_phy,
                idp_aux=self.idp_aux,
                naux=naux,
                device=device,
            )

        """ nocc_phy = n_nonint + n_int
            nocc_aux = n_nonint + n_int_aux
            n_int_aux - n_int = (naux-1)*n_int
            nocc_aux = n_nonint + (naux-1)*n_int + n_int
                     = nocc_phy + (naux-1)*n_int
        """
        self.nauxorbs = self.gGAtomic.nauxorbs

        n_int = sum([sum(self.nauxorbs[atomic_num_dict_r[an]]) for an in self.atomic_number])

        self.kinetic = Kinetic(
            atomic_number=atomic_number,
            nocc=(naux-1)*n_int+nocc,
            idp=self.idp_aux,
            kBT=kBT,
            device=device
            )
        
        self.device = device

        # dim larangian
        dl = 0
        self.basis = []
        for orb in self.gGAtomic.idp_intaux.flistnorbs: # only int aux orbital need lagrangian
            sorb = orb * 2
            dl += (sorb**2+sorb) // 2
            self.basis.append(real_hermitian_basis(sorb))

        self.lag_den_qp = torch.randn(len(atomic_number), dl, device=self.device, dtype=self.dtype) # only orbital-wise onsite block is preserved, and the redudency is removed.
    
    @property
    def LAM(self):
        # transform lag_den_qp to the LAM
        lam = torch.zeros((len(self.atomic_number), self.gGAtomic.idp_intaux.full_basis_norb*2, self.gGAtomic.idp_intaux.full_basis_norb*2), device=self.device, dtype=self.dtype)
        c_lam = 0
        c_lag = 0
        for io, orb in enumerate(self.gGAtomic.idp_intaux.flistnorbs):
            sorb = orb * 2
            n_lag = (sorb**2+sorb) // 2
            lam[:,c_lam:c_lam+sorb,c_lam:c_lam+sorb] = torch.einsum('nk,kij->nij', self.lag_den_qp[:, c_lag:c_lag+n_lag], self.basis[io])
            c_lam += sorb
            c_lag += n_lag
        
        lam.contiguous()

        return lam

    @property
    def LAM_C(self):
        return self.gGAtomic.LAM_C

    @property # beaware that the RDM can be computed from both the qp H and emb H, so we just used the prompt one as the property
    def RDM(self): # but we should keep in mind when convergence is not achieved, RDM would have several VALUE
        return self.gGAtomic.RDM # TODO: Also, despite this value is stored in gGAtomic part, 
    # it can be updated from qp H, so a update method from qp H to emb H's RDM is needed.

    @property
    def R(self):
        return self.gGAtomic.R

    @property
    def D(self):
        return self.gGAtomic.D

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        U = 0
        natom = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().shape[0]
        Rblock = torch.zeros((natom, self.idp_aux.full_basis_norb*2, self.idp_phy.full_basis_norb*2), device=self.device, dtype=self.dtype)
        # variational_density = torch.zeros((natom, self.idp_aux.full_basis_norb*2, self.idp_aux.full_basis_norb*2), device=self.device)
        variational_density = []
        for i, atomic_ansatz in enumerate(self.interaction):
            u, reduced_density, R_matrix = atomic_ansatz()
            # reduced_density has shape [norb, 2, norb, 2]
            # R_matrix has shape [aux_norb, 2, norb, 2]
            lnorb, _, rnorb, _ = R_matrix.shape
            U += u
            # copy the R value to data
            mask_left = self.idp_aux.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[i]]
            mask_left = torch.arange(len(mask_left), device=self.device)[mask_left]
            mask_right = self.idp_phy.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[i]]
            mask_right = torch.arange(len(mask_right), device=self.device)[mask_right]
            Rblock[i,mask_left.unsqueeze(1),mask_right.unsqueeze(0)] += R_matrix.reshape(lnorb*2, rnorb*2)


            # construct reduced density matrix as the format of the data
            # variational_density[i,mask_left.unsqueeze(1),mask_left.unsqueeze(0)] = reduced_density.reshape(lnorb*2, lnorb*2)
            variational_density.append(reduced_density.reshape(lnorb*2, lnorb*2))

        Rblock.contiguous()
        data[AtomicDataDict.R_MATRIX_KEY] = Rblock
        data[AtomicDataDict.INTERACTION_ENERGY_KEY] = U
        data[AtomicDataDict.VARIATIONAL_DENSITY_KEY] = torch.block_diag(*variational_density)

        data = self.hr2hk(data)
        # add the lagrangian to the kinetical part T + Lambda
        data[AtomicDataDict.HAMILTONIAN_KEY] += torch.block_diag(*self.lagrangian).unsqueeze(0)
        # compute kinetical energy and density matrix from variational single body wave function
        data = self.kinetic(data) # modified the total_energy_key and density matrix key
        
        return data
    