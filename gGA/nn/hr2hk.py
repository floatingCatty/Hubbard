import torch
from gGA.utils.constants import anglrMId, norb_dict
from typing import Tuple, Union, Dict
from gGA.data.transforms import OrbitalMapper
from gGA.data import AtomicDataDict
import re
from gGA.utils.tools import float2comlex


class GGAHR2HK(torch.nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            naux: int=1,
            spin_deg: bool=True,
            idp_phy: Union[OrbitalMapper, None]=None,
            idp_aux: Union[OrbitalMapper, None]=None,
            edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            out_field: str = AtomicDataDict.HAMILTONIAN_KEY,
            overlap: bool = False,
            device: Union[str, torch.device] = torch.device("cpu"),
            ):
        super(GGAHR2HK, self).__init__()

        self.dtype = torch.get_default_dtype()
        self.device = device
        self.overlap = overlap
        self.ctype = float2comlex(self.dtype)
        self.naux = naux
        self.spin_deg = spin_deg

        if basis is not None:
            self.idp_phy = OrbitalMapper(basis, method="e3tb", device=self.device, spin_deg=spin_deg)
            aux_basis = self.idp_phy.listnorbs.copy()
            for an in aux_basis:
                aux_basis[an] = [naux * i for i in aux_basis[an]]
            self.idp_aux = OrbitalMapper(aux_basis, method="e3tb", device=self.device, spin_deg=False)
            if idp_phy is not None:
                assert idp_phy == self.idp_phy, "The basis of idp and basis should be the same."
            if idp_aux is not None:
                assert idp_aux == self.idp_aux, "The basis of idp and basis should be the same."
        else:
            assert idp_phy is not None, "Either basis or idp should be provided."
            assert idp_phy.method == "e3tb", "The method of idp should be e3tb."
            self.idp_phy = idp_phy
            if idp_aux is None:
                # generate idp_aux
                aux_basis = {at:self.idp_phy.listnorbs[at]*naux for at in self.idp_phy.listnorbs}
                self.idp_aux = OrbitalMapper(aux_basis, method="e3tb", device=self.device, spin_deg=False)
            else:
                aux_basis = self.idp_phy.listnorbs.copy()
                for an in aux_basis:
                    aux_basis[an] = [naux * i for i in aux_basis[an]]
                assert idp_aux.listnorbs == aux_basis, "The basis of idp and basis should be the same. While left is {}, right is {}".format(idp_aux.listnorbs, aux_basis)
                assert not idp_aux.spin_deg, "The spin_deg of idp_aux should be False."
                self.idp_aux = idp_aux
        
        self.basis = self.idp_phy.basis
        self.idp_phy.get_orbpair_maps()
        self.idp_phy.get_orbpair_soc_maps()
        self.idp_aux.get_orbpair_maps()

        self.edge_field = edge_field
        self.node_field = node_field
        self.out_field = out_field

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        # construct bond wise hamiltonian block from obital pair wise node/edge features
        # we assume the edge feature have the similar format as the node feature, which is reduced from orbitals index oj-oi with j>i

        # for gGA mapping, there are two circumstances, one is spin-deg, including soc, in this case, the physical system does not have spin degree of freedom
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        orbpair_hopping = data[self.edge_field]
        orbpair_onsite = data.get(self.node_field)
        norb_phy = self.idp_phy.full_basis_norb * self.idp_phy.spin_factor
        bondwise_hopping = torch.zeros((len(orbpair_hopping), norb_phy, norb_phy), dtype=self.dtype, device=self.device)
        bondwise_hopping.to(self.device)
        bondwise_hopping.type(self.dtype)
        onsite_block = torch.zeros((len(data[AtomicDataDict.ATOM_TYPE_KEY]), norb_phy, norb_phy,), dtype=self.dtype, device=self.device)
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested:
            assert kpoints.size(0) == 1
            kpoints = kpoints[0]

        soc = data.get(AtomicDataDict.NODE_SOC_SWITCH_KEY, False)
        if isinstance(soc, torch.Tensor):
            soc = soc.all()
        if soc:
            assert self.spin_deg, "SOC is only implemented for spin-degenerate case."
            if self.overlap:
                raise NotImplementedError("Overlap is not implemented for SOC.")
            
            orbpair_soc = data[AtomicDataDict.NODE_SOC_KEY]
            soc_upup_block = torch.zeros((len(data[AtomicDataDict.ATOM_TYPE_KEY]), self.idp_phy.full_basis_norb, self.idp_phy.full_basis_norb), dtype=self.ctype, device=self.device)
            soc_updn_block = torch.zeros((len(data[AtomicDataDict.ATOM_TYPE_KEY]), self.idp_phy.full_basis_norb, self.idp_phy.full_basis_norb), dtype=self.ctype, device=self.device)

        ist = 0
        for i,io in enumerate(self.idp_phy.full_basis):
            jst = 0
            iorb = self.idp_phy.flistnorbs[i] * self.idp_phy.spin_factor
            for j,jo in enumerate(self.idp_phy.full_basis):
                jorb = self.idp_phy.flistnorbs[j] * self.idp_phy.spin_factor
                orbpair = io+"-"+jo
                
                # constructing hopping blocks
                if io == jo:
                    factor = 0.5
                else:
                    factor = 1.0

                if i <= j:
                    bondwise_hopping[:,ist:ist+iorb,jst:jst+jorb] = factor * orbpair_hopping[:,self.idp_phy.orbpair_maps[orbpair]].reshape(-1, iorb, jorb)


                # constructing onsite blocks
                if self.overlap:
                    # if iorb == jorb:
                    #     onsite_block[:, ist:ist+2*li+1, jst:jst+2*lj+1] = factor * torch.eye(2*li+1, dtype=self.dtype, device=self.device).reshape(1, 2*li+1, 2*lj+1).repeat(onsite_block.shape[0], 1, 1)
                    if i <= j:
                        onsite_block[:,ist:ist+iorb,jst:jst+jorb] = factor * orbpair_onsite[:,self.idp_phy.orbpair_maps[orbpair]].reshape(-1, iorb, jorb)
                else:
                    if i <= j:
                        onsite_block[:,ist:ist+iorb,jst:jst+jorb] = factor * orbpair_onsite[:,self.idp_phy.orbpair_maps[orbpair]].reshape(-1, iorb, jorb)
                    
                    if soc and i==j:
                        soc_updn_tmp = orbpair_soc[:,self.idp_phy.orbpair_soc_maps[orbpair]].reshape(-1, iorb, 2*jorb)
                        soc_upup_block[:,ist:ist+iorb,jst:jst+jorb] = soc_updn_tmp[:, :iorb,:jorb]
                        soc_updn_block[:,ist:ist+iorb,jst:jst+jorb] = soc_updn_tmp[:, :iorb,jorb:]
                
                jst += jorb
            ist += iorb
        
        # mapping the onsite and hopping block from physical space to auxiliary space
        if self.spin_deg:
            self.onsite_block = torch.zeros((onsite_block.shape[0], onsite_block.shape[1], 2, onsite_block.shape[2], 2), dtype=self.dtype, device=self.device)
            self.onsite_block[:,:,0,:,0] = onsite_block
            self.onsite_block[:,:,1,:,1] = onsite_block

            if soc:
                self.onsite_block[:,:,0,:,0] = self.onsite_block[:,:,0,:,0] + 0.5 * soc_upup_block
                self.onsite_block[:,:,1,:,1] = self.onsite_block[:,:,1,:,1] + 0.5 * soc_upup_block.conj()
                self.onsite_block[:,:,0,:,1] = 0.5 * soc_updn_block
                self.onsite_block[:,:,1,:,0] = 0.5 * soc_updn_block.conj()

            self.onsite_block = self.onsite_block.reshape(-1, onsite_block.shape[1]*2, onsite_block.shape[2]*2)

            self.bondwise_hopping = torch.zeros((bondwise_hopping.shape[0], bondwise_hopping.shape[1], 2, bondwise_hopping.shape[2], 2), dtype=self.dtype, device=self.device)
            self.bondwise_hopping[:,:,0,:,0] = bondwise_hopping
            self.bondwise_hopping[:,:,1,:,1] = bondwise_hopping
            self.bondwise_hopping = self.bondwise_hopping.reshape(-1, bondwise_hopping.shape[1]*2, bondwise_hopping.shape[2]*2)
        else:
            self.onsite_block = onsite_block
            self.bondwise_hopping = bondwise_hopping

        self.onsite_block = torch.bmm(data[AtomicDataDict.R_MATRIX_KEY], torch.bmm(self.onsite_block, data[AtomicDataDict.R_MATRIX_KEY].transpose(1,2)))
        self.bondwise_hopping = torch.bmm(data[AtomicDataDict.R_MATRIX_KEY][edge_index[0]], torch.bmm(self.bondwise_hopping, data[AtomicDataDict.R_MATRIX_KEY][edge_index[1]].transpose(1,2)))

        # R2K procedure can be done for all kpoint at once.
        # from now on, any spin degeneracy have been removed. All following blocks consider spin degree of freedom
        all_norb = self.idp_aux.atom_norb[data[AtomicDataDict.ATOM_TYPE_KEY]].sum() * 2
        block = torch.zeros(kpoints.shape[0], all_norb, all_norb, dtype=self.ctype, device=self.device)
        atom_id_to_indices = {}
        ist = 0
        for i, oblock in enumerate(self.onsite_block):
            mask = self.idp_aux.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[i]]
            masked_oblock = oblock[mask][:,mask]
            block[:,ist:ist+masked_oblock.shape[0],ist:ist+masked_oblock.shape[1]] = masked_oblock.unsqueeze(0)
            atom_id_to_indices[i] = slice(ist, ist+masked_oblock.shape[0])
            ist += masked_oblock.shape[0]
        

        for i, hblock in enumerate(self.bondwise_hopping):
            iatom = edge_index[0][i]
            jatom = edge_index[1][i]
            iatom_indices = atom_id_to_indices[int(iatom)]
            jatom_indices = atom_id_to_indices[int(jatom)]
            imask = self.idp_aux.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[iatom]]
            jmask = self.idp_aux.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[jatom]]
            masked_hblock = hblock[imask][:,jmask]

            block[:,iatom_indices,jatom_indices] += masked_hblock.unsqueeze(0).type_as(block) * \
                torch.exp(-1j * 2 * torch.pi * (kpoints @ data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][i])).reshape(-1,1,1)

        block = block + block.transpose(1,2).conj()
        block = block.contiguous()
        
        data[self.out_field] = block

        return data # here output hamiltonian have their spin orbital connected between each other
    