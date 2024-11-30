import torch
from gGA.data import OrbitalMapper
from gGA.data import AtomicDataDict, _keys
from typing import Tuple, Union, Dict
from gGA.nn.hr2hk import GGAHR2HK
from gGA.utils.tools import float2comlex
from gGA.utils.safe_svd import safeSVD
from torch_scatter import scatter

"""We only need to use single particle basis: |up>, |down>, which is 2 basis for each orbital to solve the problem."""

class Kinetic(object):
    def __init__(
            self,
            nocc: int,
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
        self.idx_intorb = idx_intorb
        
        assert overlap == False, "The overlap is not implemented yet."

        if basis is not None:
            self.idp_phy = OrbitalMapper(basis, method="e3tb", device=device, spin_deg=spin_deg)
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

        self.basis = self.idp_phy.basis
        self.delta_deg = delta_deg
        self.device = device

    def update(self, data: AtomicDataDict.Type, R: Dict[str, torch.Tensor], LAM: Dict[str, torch.Tensor]) -> AtomicDataDict.Type:
        if data.get(_keys.ATOMIC_NUMBERS_KEY) is not None:
            atomic_number = data[AtomicDataDict.ATOMIC_NUMBERS_KEY].flatten().clone()
            data = self.idp_phy(data)
            atom_type = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        else:
            assert data.get(AtomicDataDict.ATOM_TYPE_KEY) is not None, "The atomic number or atom type should be provided."
            atom_type = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().clone()
            atomic_number = self.idp_phy.untransform_atom(atom_type)

        phy_onsite, real_C2, tkR_C2 = self._compute(data, R, LAM)

        # solve for D
        B = torch.block_diag(*[tkR_C2[self.hr2hk.atom_id_to_indices_phy[i], self.hr2hk.atom_id_to_indices[i]] for i in range(len(atomic_number))]).T
        A = torch.block_diag(*[real_C2[self.hr2hk.atom_id_to_indices[i], self.hr2hk.atom_id_to_indices[i]] for i in range(len(atomic_number))])
        A = symsqrt(A @ (torch.eye(A.shape[0], device=self.device) - A)).T

        D = torch.linalg.solve(A, B)
        out_D = {sym: [] for sym in self.idp_phy.type_names}
        out_RDM = {sym: [] for sym in self.idp_phy.type_names}
        # split D and RDM
        for i, an in enumerate(atomic_number):
            out_D[self.idp_phy.type_names[atom_type[i]]].append(D[self.hr2hk.atom_id_to_indices[i], self.hr2hk.atom_id_to_indices_phy[i]])
            out_RDM[self.idp_phy.type_names[atom_type[i]]].append(real_C2[self.hr2hk.atom_id_to_indices[i], self.hr2hk.atom_id_to_indices[i]])
        
        for sym in self.idp_phy.type_names:
            out_D[sym] = torch.stack(out_D[sym])
            out_RDM[sym] = torch.stack(out_RDM[sym])

        return phy_onsite, out_D, out_RDM

    def compute_RDM(self, data: AtomicDataDict.Type, R: Dict[str, torch.Tensor], LAM: Dict[str, torch.Tensor]):
        if data.get(_keys.ATOMIC_NUMBERS_KEY) is not None:
            atomic_number = data[AtomicDataDict.ATOMIC_NUMBERS_KEY].flatten().clone()
            data = self.idp_phy(data)
            atom_type = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        else:
            assert data.get(AtomicDataDict.ATOM_TYPE_KEY) is not None, "The atomic number or atom type should be provided."
            atom_type = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().clone()
            atomic_number = self.idp_phy.untransform_atom(atom_type)

        _, real_C2, _ = self._compute(data, R, LAM)
        out_RDM = {sym: [] for sym in self.idp_phy.type_names}
        # split D and RDM
        for i, an in enumerate(atomic_number):
            out_RDM[self.idp_phy.type_names[atom_type[i]]].append(real_C2[self.hr2hk.atom_id_to_indices[i], self.hr2hk.atom_id_to_indices[i]])
        
        for sym in self.idp_phy.type_names:
            out_RDM[sym] = torch.stack(out_RDM[sym])

        return out_RDM

    def _compute(self, data: AtomicDataDict.Type, R: Dict[str, torch.Tensor], LAM: Dict[str, torch.Tensor]):
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested:
            assert kpoints.size(0) == 1
            kpoints = kpoints[0]

        # construct LAM
        phy_onsite, H, tkR = self.hr2hk(data, R, LAM)

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

        scatter_index = torch.arange(nk).reshape(-1,1).repeat(1, norb).reshape(-1)
        scatter_index = scatter_index[mask.flatten()]
        vec_m = eigvec[mask]
        real_C2 = torch.einsum("ni, nj->nij", vec_m.conj(), vec_m) # nstates, norb*2, norb*2
        real_C2 = scatter(real_C2, scatter_index, dim=0, reduce="sum")

        if mask_left is not None:
            scatter_index = torch.arange(nk).reshape(-1,1).repeat(1, norb).reshape(-1)
            scatter_index = scatter_index[mask_left]
            vec_ml = eigvec[mask_left]
            real_C2 += torch.scatter((state_left/ndegstates) * torch.einsum("ni, nj->nij", vec_ml.conj(), vec_ml), scatter_index, dim=0, reduce="sum")
            # the factor need to be carefully revised

        tkR_C2 = torch.bmm(tkR, real_C2).sum(0) / nk
        # check imaginary part
        assert tkR_C2.imag.abs().max() < 1e-8, "The imaginary part of tkR_C2 is not zero."
        assert real_C2.imag.abs().max() < 1e-8, "The imaginary part of real_C2 is not zero."

        tkR_C2 = tkR_C2.real
        real_C2 = real_C2.real.sum(0) / nk

        return phy_onsite, real_C2, tkR_C2


        
    


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
