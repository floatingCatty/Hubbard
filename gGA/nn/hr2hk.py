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
            spin_deg: bool=True,
            idp_phy: Union[OrbitalMapper, None]=None,
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
        self.spin_deg = spin_deg

        if basis is not None:
            self.idp_phy = OrbitalMapper(basis, method="e3tb", device=self.device, spin_deg=spin_deg)
            if idp_phy is not None:
                assert idp_phy == self.idp_phy, "The basis of idp and basis should be the same."
        else:
            assert idp_phy is not None, "Either basis or idp should be provided."
            assert idp_phy.method == "e3tb", "The method of idp should be e3tb."
            self.idp_phy = idp_phy

        
        self.basis = self.idp_phy.basis
        self.idp_phy.get_orbpair_maps()
        self.idp_phy.get_orbpair_soc_maps()

        self.edge_field = edge_field
        self.node_field = node_field
        self.out_field = out_field

    def forward(self, data: AtomicDataDict.Type, R: Dict[str, torch.Tensor], LAM: Dict[str, torch.Tensor]) -> AtomicDataDict.Type:

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

        #       constructing phy blocks
        if self.spin_deg:
            onsite_block = torch.stack([onsite_block, torch.zeros_like(onsite_block), torch.zeros_like(onsite_block), onsite_block], dim=-1).permute(0,1,3,2,4)

            if soc:
                onsite_block[:,:,0,:,0] = onsite_block[:,:,0,:,0] + 0.5 * soc_upup_block
                onsite_block[:,:,1,:,1] = onsite_block[:,:,1,:,1] + 0.5 * soc_upup_block.conj()
                onsite_block[:,:,0,:,1] = 0.5 * soc_updn_block
                onsite_block[:,:,1,:,0] = 0.5 * soc_updn_block.conj()

            onsite_block = onsite_block.reshape(-1, onsite_block.shape[1]*2, onsite_block.shape[2]*2)

            bondwise_hopping = torch.stack([bondwise_hopping, torch.zeros_like(bondwise_hopping), torch.zeros_like(bondwise_hopping), bondwise_hopping], dim=-1).permute(0,1,3,2,4)
            bondwise_hopping = bondwise_hopping.reshape(-1, bondwise_hopping.shape[1]*2, bondwise_hopping.shape[2]*2)

        self.onsite_block = {}
        self.bondwise_hopping = {}
        onsite_tkR = {}
        hopping_tkR = {}
        norb_aux = 0
        for sym, at in self.idp_phy.chemical_symbol_to_type.items():
            mask = self.idp_phy.mask_to_basis[at]
            atmask = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().eq(at)
            self.onsite_block[sym] = onsite_block[atmask][:,mask][:,:,mask]
            onsite_tkR[sym] = torch.bmm(self.onsite_block[sym], R[sym].transpose(1,2))
            self.onsite_block[sym] = torch.bmm(torch.bmm(R[sym], self.onsite_block[sym]), R[sym].transpose(1,2))
            norb_aux += self.onsite_block[sym].shape[1] * self.onsite_block[sym].shape[0]
        
        edge_atom_type1 = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[edge_index[0]]
        edge_atom_type2 = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[edge_index[1]]
        for bsym, bt in self.idp_phy.bond_to_type.items():
            asym, asym = bsym.split("-")
            at1, at2 = self.idp_phy.chemical_symbol_to_type[asym], self.idp_phy.chemical_symbol_to_type[bsym]
            mask1 = self.idp_phy.mask_to_basis[at1]
            mask2 = self.idp_phy.mask_to_basis[at2]
            btmask = data[AtomicDataDict.EDGE_TYPE_KEY].flatten().eq(bt)
            self.bondwise_hopping[bsym] = bondwise_hopping[btmask][:,mask1][:,:,mask2]
            index1 = torch.cumsum(edge_atom_type1.eq(at), dim=0)[edge_index[0]] - 1
            index2 = torch.cumsum(edge_atom_type2.eq(at), dim=0)[edge_index[1]] - 1
            hopping_tkR[bsym] = torch.bmm(self.bondwise_hopping[bsym], R[at2][index2].transpose(1,2))
            self.bondwise_hopping[bsym] = torch.bmm(torch.bmm(R[at1][index1], self.bondwise_hopping[bsym]), R[at2][index2].transpose(1,2))
        
        # R2K procedure can be done for all kpoint at once.
        # from now on, any spin degeneracy have been removed. All following blocks consider spin degree of freedom
        block = torch.zeros(kpoints.shape[0], norb_aux, norb_aux, dtype=self.ctype, device=self.device)
        tkR = torch.zeros(kpoints.shape[0], self.idp_phy.atom_norb[data[AtomicDataDict.ATOM_TYPE_KEY]].sum(), norb_aux, dtype=self.ctype, device=self.device)
        atom_id_to_indices = {}
        atom_id_to_indices_phy = {}
        ist = 0
        ist_tkR = 0
        type_count = [0] * len(self.idp_phy.type_names)
        for i, at in enumerate(data[AtomicDataDict.ATOM_TYPE_KEY].flatten()):
            idx = type_count[at]
            sym = self.idp_phy.type_names[at]
            oblock = self.onsite_block[sym][idx] + LAM[sym][idx]
            oblock_tkR = onsite_tkR[sym][idx]
            block[:,ist:ist+oblock.shape[0],ist:ist+oblock.shape[1]] = oblock.unsqueeze(0)
            tkR[:, ist_tkR:ist_tkR+oblock_tkR.shape[0], ist:ist+oblock.shape[1]] = oblock_tkR.unsqueeze(0)
            atom_id_to_indices[i] = slice(ist, ist+oblock.shape[0])
            atom_id_to_indices_phy[i] = slice(ist_tkR, ist_tkR+oblock_tkR.shape[0])
            ist += oblock.shape[0]
            ist_tkR += oblock_tkR.shape[0]
            type_count[at] += 1
        

        for bsym, btype in self.idp_phy.bond_to_type.items():
            for i, edge in enumerate(edge_index.T[data[AtomicDataDict.EDGE_TYPE_KEY].flatten().eq(btype)]):
                iatom = edge[0]
                jatom = edge[1]
                iatom_indices = atom_id_to_indices[int(iatom)]
                iatom_indices_phy = atom_id_to_indices_phy[int(iatom)]
                jatom_indices = atom_id_to_indices[int(jatom)]
                hblock = self.bondwise_hopping[bsym][i]
                hblock_tkR = hopping_tkR[bsym][i]

                block[:,iatom_indices,jatom_indices] += hblock.unsqueeze(0).type_as(block) * \
                    torch.exp(-1j * 2 * torch.pi * (kpoints @ data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][i])).reshape(-1,1,1)
                tkR[:, iatom_indices_phy, jatom_indices] += hblock_tkR.unsqueeze(0).type_as(tkR) * \
                    torch.exp(-1j * 2 * torch.pi * (kpoints @ data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][i])).reshape(-1,1,1)

        block = block + block.transpose(1,2).conj()
        block = block.contiguous()

        return block, tkR # here output hamiltonian have their spin orbital connected between each other
    