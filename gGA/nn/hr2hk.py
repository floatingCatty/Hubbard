import torch
from gGA.utils.constants import anglrMId, norb_dict
from typing import Tuple, Union, Dict
from gGA.data.transforms import OrbitalMapper
from gGA.data import AtomicDataDict
from gGA.utils.tools import float2comlex


class GGAHR2HK(torch.nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            spin_deg: bool=True,
            idp_phy: Union[OrbitalMapper, None]=None,
            idx_intorb: Dict[str, list]=None,
            edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            overlap: bool = False,
            device: Union[str, torch.device] = torch.device("cpu")
            ):
        
        super(GGAHR2HK, self).__init__()

        self.dtype = torch.get_default_dtype()
        self.device = device
        self.overlap = overlap
        self.ctype = float2comlex(self.dtype)
        self.spin_deg = spin_deg
        self.idx_intorb = idx_intorb

        # check whether there are interacting orbitals
        self.interaction = False
        if self.idx_intorb is not None and len(self.idx_intorb) > 0:
            for sym, orbs in self.idx_intorb.items():
                if len(orbs) > 0:
                    self.interaction = True
                    break


        if basis is not None:
            self.idp_phy = OrbitalMapper(basis, method="e3tb", device=self.device, spin_deg=spin_deg)
            if idp_phy is not None:
                assert idp_phy == self.idp_phy, "The basis of idp and basis should be the same."
        else:
            assert idp_phy is not None, "Either basis or idp should be provided."
            assert idp_phy.method == "e3tb", "The method of idp should be e3tb."
            self.idp_phy = idp_phy

        # get mask_to_basis incase that spin is neglected
        mask_to_basis = self.idp_phy.mask_to_basis.clone()
        if self.spin_deg:
            mask_to_basis = mask_to_basis.unsqueeze(-1).repeat(1,1,2).reshape(mask_to_basis.shape[0], -1)
        self.mask_to_basis = mask_to_basis

        if self.interaction:
            # construct the project basis
            self.map_noint = {}
            for sym, orbs in self.idp_phy.basis.items():
                n_int = len(self.idx_intorb.get(sym,[]))
                self.map_noint[sym] = torch.zeros(len(orbs)-n_int, sum(self.idp_phy.listnorbs[sym])*2, dtype=torch.bool, device=self.device)
                count = 0
                for io, orb in enumerate(orbs):
                    if self.idx_intorb.get(sym) is None or not io in self.idx_intorb[sym]:
                        norb = self.idp_phy.listnorbs[sym][io]
                        snorb = sum(self.idp_phy.listnorbs[sym][:io])
                        self.map_noint[sym][count][snorb*2:snorb*2+norb*2] = True
                        count += 1
                assert count == self.map_noint[sym].shape[0]

        
        self.basis = self.idp_phy.basis
        self.idp_phy.get_orbpair_maps()
        self.idp_phy.get_orbpair_soc_maps()

        self.edge_field = edge_field
        self.node_field = node_field

    def forward(self, data: AtomicDataDict.Type, R: Dict[str, torch.Tensor]=None, LAM: Dict[str, torch.Tensor]=None, TMAT: Dict[str, torch.Tensor]=None) -> AtomicDataDict.Type:

        # construct bond wise hamiltonian block from obital pair wise node/edge features
        # we assume the edge feature have the similar format as the node feature, which is reduced from orbitals index oj-oi with j>i

        # for gGA mapping, there are two circumstances, one is spin-deg, including soc, in this case, the physical system does not have spin degree of freedom
        if self.interaction:
            assert R is not None, "R must be provided if there exist any interacting orbitals"

        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        orbpair_hopping = data[self.edge_field]
        orbpair_onsite = data.get(self.node_field)
        norb_phy = self.idp_phy.full_basis_norb
        bondwise_hopping = torch.zeros((len(orbpair_hopping), norb_phy, norb_phy), dtype=self.dtype, device=self.device)
        bondwise_hopping.to(self.device)
        bondwise_hopping.type(self.dtype)
        onsite_block = torch.zeros((len(data[AtomicDataDict.ATOM_TYPE_KEY]), norb_phy, norb_phy,), dtype=self.dtype, device=self.device)
        if self.interaction and not self.overlap:
            phy_onsite = torch.zeros_like(onsite_block)
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
                    if i < j:
                        onsite_block[:,ist:ist+iorb,jst:jst+jorb] = factor * orbpair_onsite[:,self.idp_phy.orbpair_maps[orbpair]].reshape(-1, iorb, jorb)
                    elif i == j: # this step remove also the diagonal element of onsite_block of non-interacting orbitals, needed to added it back
                        if self.interaction:
                            phy_onsite[:,ist:ist+iorb,jst:jst+jorb] = orbpair_onsite[:,self.idp_phy.orbpair_maps[orbpair]].reshape(-1, iorb, jorb)
                        else:
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
            shape = list(onsite_block.shape)
            onsite_block = torch.stack([onsite_block, torch.zeros_like(onsite_block), torch.zeros_like(onsite_block), onsite_block], dim=-1).reshape(shape+[2,2]).permute(0,1,3,2,4)
            if self.interaction and not self.overlap:
                phy_onsite = torch.stack([phy_onsite, torch.zeros_like(phy_onsite), torch.zeros_like(phy_onsite), phy_onsite], dim=-1).reshape(shape+[2,2]).permute(0,1,3,2,4)
                phy_onsite = phy_onsite.reshape(-1, phy_onsite.shape[1]*2, phy_onsite.shape[3]*2)

            if soc:
                onsite_block[:,:,0,:,0] = onsite_block[:,:,0,:,0] + 0.5 * soc_upup_block
                onsite_block[:,:,1,:,1] = onsite_block[:,:,1,:,1] + 0.5 * soc_upup_block.conj()
                onsite_block[:,:,0,:,1] = 0.5 * soc_updn_block
                onsite_block[:,:,1,:,0] = 0.5 * soc_updn_block.conj()

            onsite_block = onsite_block.reshape(-1, onsite_block.shape[1]*2, onsite_block.shape[3]*2)
            shape = list(bondwise_hopping.shape)
            bondwise_hopping = torch.stack([bondwise_hopping, torch.zeros_like(bondwise_hopping), torch.zeros_like(bondwise_hopping), bondwise_hopping], dim=-1).reshape(shape+[2,2]).permute(0,1,3,2,4)
            bondwise_hopping = bondwise_hopping.reshape(-1, bondwise_hopping.shape[1]*2, bondwise_hopping.shape[3]*2)

        self.phy_onsite = {}
        self.onsite_block = {}
        self.bondwise_hopping = {}
        onsite_tkR = {}
        hopping_tkR = {}
        norb_aux = 0
        for sym, at in self.idp_phy.chemical_symbol_to_type.items():
            mask = self.mask_to_basis[at]
            atmask = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().eq(at)
            self.onsite_block[sym] = onsite_block[atmask][:,mask][:,:,mask]

            # first step, add back the non-interacting terms in hamiltonian
            if self.interaction and not self.overlap:
                self.phy_onsite[sym] = phy_onsite[atmask][:,mask][:,:,mask]

                non_int_mask = self.map_noint[sym]
                non_int_mask = (non_int_mask[:,:,None] * non_int_mask[:,None,:]).sum(0).bool()
                self.onsite_block[sym][:,non_int_mask] += 0.5 * self.phy_onsite[sym][:, non_int_mask]

                onsite_tkR[sym] = self.onsite_block[sym].clone()
            
            if self.interaction and sym in self.idx_intorb.keys(): # so R must present
                self.onsite_block[sym] = torch.bmm(torch.bmm(R[sym], self.onsite_block[sym].type_as(R[sym])), R[sym].transpose(1,2).conj())

            norb_aux += self.onsite_block[sym].shape[1] * self.onsite_block[sym].shape[0]
        
        for bsym, bt in self.idp_phy.bond_to_type.items():
            sym1, sym2 = bsym.split("-")
            at1, at2 = self.idp_phy.chemical_symbol_to_type[sym1], self.idp_phy.chemical_symbol_to_type[sym2]
            mask1 = self.mask_to_basis[at1]
            mask2 = self.mask_to_basis[at2]
            btmask = data[AtomicDataDict.EDGE_TYPE_KEY].flatten().eq(bt)
            self.bondwise_hopping[bsym] = bondwise_hopping[btmask][:,mask1][:,:,mask2]
            if self.interaction and not self.overlap:
                hopping_tkR[bsym] = bondwise_hopping[btmask][:,mask1][:,:,mask2]

            index1 = torch.cumsum(atom_types.eq(at), dim=0)[edge_index[0][btmask]] - 1
            index2 = torch.cumsum(atom_types.eq(at), dim=0)[edge_index[1][btmask]] - 1 # Here hopping and onsite have not include the conjugated part, so tkR calculation is wrong!
            if self.interaction:
                if sym1 in self.idx_intorb:
                    self.bondwise_hopping[bsym] = torch.bmm(R[sym1][index1], self.bondwise_hopping[bsym].type_as(R[sym1]))
                if sym2 in self.idx_intorb:
                    self.bondwise_hopping[bsym] = torch.bmm(self.bondwise_hopping[bsym].type_as(R[sym2]), R[sym2][index2].transpose(1,2).conj())
        
        # R2K procedure can be done for all kpoint at once.
        # from now on, any spin degeneracy have been removed. All following blocks consider spin degree of freedom
        block = torch.zeros(kpoints.shape[0], norb_aux, norb_aux, dtype=self.ctype, device=self.device)
        norb_phy = self.idp_phy.atom_norb[data[AtomicDataDict.ATOM_TYPE_KEY]].sum()
        if self.spin_deg:
            norb_phy *= 2
        if self.interaction and not self.overlap:
            tkR = torch.zeros(kpoints.shape[0], norb_phy, norb_phy, dtype=self.ctype, device=self.device)
        else:
            tkR = None

        atom_id_to_indices = {}
        atom_id_to_indices_phy = {}
        ist = 0
        ist_tkR = 0
        type_count = [0] * len(self.idp_phy.type_names)
        Rs = []
        for i, at in enumerate(data[AtomicDataDict.ATOM_TYPE_KEY].flatten()):
            idx = type_count[at]
            sym = self.idp_phy.type_names[at]
            oblock = self.onsite_block[sym][idx]
            if LAM is not None:
                if self.overlap: 
                    raise RuntimeError("LAM cannot be added on the Overlap!")
                elif not self.interaction:
                    raise RuntimeError("LAM cannot be added on the Hamiltonian without interacting orbitals!")
                elif sym in self.idx_intorb.keys():
                    oblock = oblock + 0.5 * LAM[sym][idx]
            
            block[:,ist:ist+oblock.shape[0],ist:ist+oblock.shape[1]] = oblock.unsqueeze(0)
            atom_id_to_indices[i] = slice(ist, ist+oblock.shape[0])
            
            if self.interaction and not self.overlap:
                oblock_tkR = onsite_tkR[sym][idx]
                tkR[:, ist_tkR:ist_tkR+oblock_tkR.shape[0], ist_tkR:ist_tkR+oblock_tkR.shape[1]] = oblock_tkR.unsqueeze(0)
                atom_id_to_indices_phy[i] = slice(ist_tkR, ist_tkR+oblock_tkR.shape[0])
                ist_tkR += oblock_tkR.shape[0]
                if sym in self.idx_intorb.keys():
                    Rs.append(R[sym][idx].H)
                else:
                    Rs.append(torch.eye(oblock_tkR.shape[1], device=self.device, dtype=self.dtype))
            
            ist += oblock.shape[0]

            type_count[at] += 1


        for bsym, btype in self.idp_phy.bond_to_type.items():
            bmask = data[AtomicDataDict.EDGE_TYPE_KEY].flatten().eq(btype)
            shifts_vec = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][bmask]
            for i, edge in enumerate(edge_index.T[bmask]):
                iatom = edge[0]
                jatom = edge[1]
                iatom_indices = atom_id_to_indices[int(iatom)]
                jatom_indices = atom_id_to_indices[int(jatom)]
                hblock = self.bondwise_hopping[bsym][i]
                block[:,iatom_indices,jatom_indices] += hblock.unsqueeze(0).type_as(block) * \
                    torch.exp(-1j * 2 * torch.pi * (kpoints @ shifts_vec[i])).reshape(-1,1,1)
                
                if self.interaction and not self.overlap:
                    hblock_tkR = hopping_tkR[bsym][i]
                    iatom_indices_phy = atom_id_to_indices_phy[int(iatom)]
                    j_atom_indices_phy = atom_id_to_indices_phy[int(jatom)]
                    tkR[:, iatom_indices_phy, j_atom_indices_phy] += hblock_tkR.unsqueeze(0).type_as(tkR) * \
                        torch.exp(-1j * 2 * torch.pi * (kpoints @ shifts_vec[i])).reshape(-1,1,1)

        block = block + block.transpose(1,2).conj()
        block = block.contiguous()

        if self.interaction and not self.overlap:
            tkR = tkR + tkR.transpose(1,2).conj()
            tkR = tkR.contiguous()
            tkR = tkR @ torch.block_diag(*Rs).unsqueeze(0).type_as(tkR)

        return self.phy_onsite, block, tkR # here output hamiltonian have their spin orbital connected between each other