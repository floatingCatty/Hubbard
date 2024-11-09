"""
Input: atomicdata class

output: atomic data class with new node features as R, D
"""
import torch
import torch.nn as nn
from gGA.data import OrbitalMapper
from gGA.nn.ansatz import gGASingleOrb, gGAMultiOrb
from gGA.utils.constants import atomic_num_dict_r
from gGA.data import AtomicDataDict
from gGA.nn.hr2hk import GGAHR2HK
from gGA.nn.kinetics import Kinetic
from typing import Union, Dict

class GostGutzwiller(nn.Module):
    def __init__(
            self, 
            atomic_number, 
            nocc, 
            basis, 
            naux, 
            nk,
            Hint_params, 
            spin_deg=True, 
            device: Union[str, torch.device] = torch.device("cpu"),
            kBT=0.025,
            ):
        super(GostGutzwiller, self).__init__()
        self.basis = basis
        self.atomic_number = atomic_number
        self.spin_deg = spin_deg
        self.idp_phy = OrbitalMapper(basis=basis, device=device, spin_deg=False)
        self.dtype = torch.get_default_dtype()
        self.naux = naux
        # generate the idp for gost system
        listnorbs = self.idp_phy.listnorbs.copy()
        self.interaction = nn.ModuleList([
            gGAMultiOrb(
                norbs=listnorbs[atomic_num_dict_r[int(atomic_number[an])]], 
                naux=naux, 
                Hint_params=Hint_params[int(an)],
                device=device
                ) 
                for an in range(len(atomic_number))
            ])
        for an in listnorbs:
            listnorbs[an] = [naux * i for i in listnorbs[an]]
        self.idp_aux = OrbitalMapper(basis=listnorbs, device=device, spin_deg=False)
        

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

        self.totalnorb = int(self.idp_phy.atom_norb[self.idp_phy.transform(atomic_number).flatten()].sum())
        self.kinetic = Kinetic(
            atomic_number=atomic_number,
            nocc=self.totalnorb*(naux-1)+nocc,
            idp=self.idp_aux,
            nk=nk,
            kBT=kBT,
            device=device
            )
        
        self.device = device

        self.lagrangian = []
        for inorb in self.idp_phy.atom_norb[self.idp_phy.transform(atomic_number).flatten()]:
            n = torch.randn((inorb*2*self.naux, inorb*2*self.naux), device=self.device, dtype=self.dtype)
            self.lagrangian.append(
                (n + n.transpose(0,1)).abs() / (2**0.5)
            )
        self.lagrangian = nn.ParameterList(self.lagrangian)
        # self.lagrangianR = nn.Parameter(
        #     torch.randn((len(self.atomic_number), self.idp_aux.full_basis_norb*2, self.idp_phy.full_basis_norb*2), device=self.device, dtype=self.dtype).abs()
        # )

        # lagrangianR = nn.Parameter(torch.randn(
        #     (len(self.atomic_number), self.totalnorb*2*self.naux, self.totalnorb*2), 
        #     device=self.device, dtype=self.dtype).abs())

        # self.lagrangian.append(lagrangianR)
        # for p in self.lagrangian:
        #     p.requires_grad = False

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
        # data[AtomicDataDict.HAMILTONIAN_KEY] += torch.block_diag(*self.lagrangian).unsqueeze(0)
        # compute kinetical energy and density matrix from variational single body wave function
        data = self.kinetic(data) # modified the total_energy_key and density matrix key
        
        return data

        


