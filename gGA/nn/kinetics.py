import torch
from gGA.data import OrbitalMapper
from gGA.data import AtomicDataDict, _keys
from typing import Tuple, Union, Dict
from gGA.nn.hr2hk import GGAHR2HK
from gGA.utils.tools import float2comlex
from gGA.utils.constants import atomic_num_dict_r
from torch_scatter import scatter
import copy

"""We only need to use single particle basis: |up>, |down>, which is 2 basis for each orbital to solve the problem."""

class Kinetic(object):
    def __init__(
            self,
            nocc: int,
            naux: int,
            idp_phy: Union[OrbitalMapper, None]=None, 
            basis: Dict[str, Union[str, list]]=None,
            idx_intorb: Dict[str, list]=None,
            kBT: float = 0.001, # 0.0257,
            device: Union[str, torch.device] = torch.device("cpu"),
            delta_deg=1e-4,
            overlap: bool = False,
            soc: int = False, # control whether spin-orbital coupling is included, if it is true, nspin will be set as 4.
            nspin: int = 1, # control the spin orbitals, 1 for spin degenerate, 2 for spin collinear polarization, 4 for spin non-collinear polarization
            ):
        
        super(Kinetic, self).__init__()
        self.ctype = float2comlex(torch.get_default_dtype())
        self.nocc = nocc
        self.kBT = kBT
        self.idx_intorb = idx_intorb
        self.naux = naux
        self.soc = soc
        self.spin_deg = nspin <= 1
        self.nspin = nspin

        if soc:
            self.nspin = 4
        else:
            self.nspin = nspin
        
        assert overlap == False, "The overlap is not implemented yet."

        if basis is not None:
            self.idp_phy = OrbitalMapper(basis, method="e3tb", device=device, spin_deg=self.spin_deg)
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
            spin_deg=self.spin_deg,
            device=device
        )

        self.basis = self.idp_phy.basis
        self.delta_deg = delta_deg
        self.device = device

        nauxorbs = copy.deepcopy(self.idp_phy.listnorbs) # {'C': [1, 3], 'Si': [1, 1, 3, 3, 5]}
        for sym in nauxorbs:
            for i in self.idx_intorb[sym]:
                nauxorbs[sym][i] = nauxorbs[sym][i] * naux
        self.nauxorbs = nauxorbs

        start_int_orbs = {sym:[] for sym in self.idp_phy.type_names}
        start_phy_orbs = {sym:[] for sym in self.idp_phy.type_names}
        for sym in self.idp_phy.type_names:
            for i in self.idx_intorb[sym]:
                start_int_orbs[sym].append(sum(nauxorbs[sym][:i])*2)
                start_phy_orbs[sym].append(sum(self.idp_phy.listnorbs[sym][:i]))
        self.start_int_orbs = start_int_orbs
        self.start_phy_orbs = start_phy_orbs


    def update(self, data: AtomicDataDict.Type, R: Dict[str, torch.Tensor], LAM: Dict[str, torch.Tensor]) -> AtomicDataDict.Type:
        if data.get(_keys.ATOMIC_NUMBERS_KEY) is not None:
            atomic_number = data[AtomicDataDict.ATOMIC_NUMBERS_KEY].flatten().clone()
            data = self.idp_phy(data)
            atom_type = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        else:
            assert data.get(AtomicDataDict.ATOM_TYPE_KEY) is not None, "The atomic number or atom type should be provided."
            atom_type = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().clone()
            atomic_number = self.idp_phy.untransform_atom(atom_type)

        phy_onsite, real_C2, tkR_C2, E_fermi = self._compute(data, R, LAM)

        # solve for D
        B = []
        A = []
        out_RDM = {sym: [] for sym in self.idp_phy.type_names}
        for i, an in enumerate(atomic_number):
            sym = atomic_num_dict_r[int(an)]
            iatkC2 = tkR_C2[self.hr2hk.atom_id_to_indices_phy[i], self.hr2hk.atom_id_to_indices[i]]
            iaC2 = real_C2[self.hr2hk.atom_id_to_indices[i], self.hr2hk.atom_id_to_indices[i]]
            sel_iaC2 = [torch.eye(j*2, device=self.device) for j in self.idp_phy.listnorbs[sym]]
            sel_iatkC2 = [torch.eye(j*2, device=self.device) for j in self.idp_phy.listnorbs[sym]]

            for jx, j in enumerate(self.idx_intorb[sym]):
                sel_iaC2[j] = iaC2[self.start_int_orbs[sym][jx]:self.start_int_orbs[sym][jx]+self.nauxorbs[sym][j]*2, 
                                    self.start_int_orbs[sym][jx]:self.start_int_orbs[sym][jx]+self.nauxorbs[sym][j]*2]
                sel_iatkC2[j] = iatkC2[self.start_phy_orbs[sym][jx]:self.start_phy_orbs[sym][jx]+self.idp_phy.listnorbs[sym][j]*2,
                                       self.start_int_orbs[sym][jx]:self.start_int_orbs[sym][jx]+self.nauxorbs[sym][j]*2]
            A.append(torch.block_diag(*sel_iaC2))
            B.append(torch.block_diag(*sel_iatkC2))
            out_RDM[self.idp_phy.type_names[atom_type[i]]].append(torch.block_diag(*sel_iaC2))
        A = torch.block_diag(*A)
        B = torch.block_diag(*B).T

        A = symsqrt(A @ (torch.eye(A.shape[0], device=self.device) - A))
        try:
            print("DM_kin: ", torch.linalg.eigvalsh(real_C2))
            D = torch.linalg.solve(A, B)
        except:
            import matplotlib.pyplot as plt
            plt.matshow(real_C2.detach(), cmap="bwr", vmax=1.0, vmin=-1.0)
            plt.show()
            raise RuntimeError("The solve does not converge.")
        
        out_D = {sym: [] for sym in self.idp_phy.type_names}

        # split D and RDM
        for i, an in enumerate(atomic_number):
            sym = atomic_num_dict_r[int(an)]
            iaD = D[self.hr2hk.atom_id_to_indices[i], self.hr2hk.atom_id_to_indices_phy[i]]
            sel_iaD = [torch.eye(j*2, device=self.device) for j in self.idp_phy.listnorbs[sym]]
            for jx, j in enumerate(self.idx_intorb[sym]):
                sel_iaD[j] = iaD[self.start_int_orbs[sym][jx]:self.start_int_orbs[sym][jx]+self.nauxorbs[sym][j]*2, 
                                 self.start_phy_orbs[sym][jx]:self.start_phy_orbs[sym][jx]+self.idp_phy.listnorbs[sym][j]*2]
                
            out_D[self.idp_phy.type_names[atom_type[i]]].append(torch.block_diag(*sel_iaD))
        
        for sym in self.idp_phy.type_names:
            out_D[sym] = torch.stack(out_D[sym])
            out_RDM[sym] = torch.stack(out_RDM[sym])

        return phy_onsite, out_D, out_RDM, E_fermi

    def compute_RDM(self, data: AtomicDataDict.Type, R: Dict[str, torch.Tensor], LAM: Dict[str, torch.Tensor]):
        if data.get(_keys.ATOMIC_NUMBERS_KEY) is not None:
            atomic_number = data[AtomicDataDict.ATOMIC_NUMBERS_KEY].flatten().clone()
            data = self.idp_phy(data)
            atom_type = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        else:
            assert data.get(AtomicDataDict.ATOM_TYPE_KEY) is not None, "The atomic number or atom type should be provided."
            atom_type = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().clone()
            atomic_number = self.idp_phy.untransform_atom(atom_type)

        _, real_C2, _, _ = self._compute(data, R, LAM)
        out_RDM = {sym: [] for sym in self.idp_phy.type_names}
        # split D and RDM
        for i, an in enumerate(atomic_number):
            sym = atomic_num_dict_r[int(an)]
            iaRDM = real_C2[self.hr2hk.atom_id_to_indices[i], self.hr2hk.atom_id_to_indices[i]]
            sel_iaRD = [torch.eye(j*2, device=self.device) for j in self.idp_phy.listnorbs[sym]]
            for jx, j in enumerate(self.idx_intorb[sym]):
                sel_iaRD[j] = iaRDM[self.start_int_orbs[sym][jx]:self.start_int_orbs[sym][jx]+self.nauxorbs[sym][j]*2, 
                                    self.start_int_orbs[sym][jx]:self.start_int_orbs[sym][jx]+self.nauxorbs[sym][j]*2]
            out_RDM[self.idp_phy.type_names[atom_type[i]]].append(torch.block_diag(*sel_iaRD))
        
        for sym in self.idp_phy.type_names:
            out_RDM[sym] = torch.stack(out_RDM[sym])

        return out_RDM

    def _compute(self, data: AtomicDataDict.Type, R: Dict[str, torch.Tensor], LAM: Dict[str, torch.Tensor], E_fermi: float=None):
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
        
        if E_fermi is None:
            E_fermi = torch.sort(eigval.flatten())[0][self.nocc*nk-1:self.nocc*nk+1].sum() * 0.5
            E_fermi = find_E_fermi(Hks=H, nocc=self.nocc, kBT=self.kBT, E_fermi0=E_fermi.item())
            print("Find E_fermi: {:.4f}".format(E_fermi))

        factor = fermi_dirac(eigval, E_fermi, self.kBT).detach() # [nk, norb]
        mask = factor > 1e-10 # [nk, norb]
        scatter_index = torch.arange(nk).reshape(-1,1).repeat(1, norb).reshape(-1) # [nk*norb]
        scatter_index = scatter_index[mask.flatten()]
        vec_m = (eigvec * factor.unsqueeze(-1))[mask]
        real_C2 = torch.einsum("ni, nj->nij", vec_m.conj(), eigvec[mask]) # nstates, norb*2, norb*2
        real_C2 = scatter(real_C2, scatter_index, dim=0, reduce="sum", dim_size=nk)

        tkR_C2 = torch.bmm(tkR, real_C2.permute(0,2,1)).sum(0) / nk
        # check imaginary part
        assert tkR_C2.imag.abs().max() < 1e-8, "The imaginary part of tkR_C2 {} is not zero.".format(tkR_C2.imag.abs().max())
        assert real_C2.imag.abs().max() < 1e-8, "The imaginary part of real_C2 {} is not zero.".format(real_C2.imag.abs().max())

        tkR_C2 = tkR_C2.real
        real_C2 = real_C2.real.sum(0) / nk

        return phy_onsite, real_C2, tkR_C2, E_fermi



def compute_nocc(Hks, E_fermi: float, kBT: float):
    """
        Hks: [nk, norb, norb],
    """
    eigval = torch.linalg.eigvalsh(Hks)
    counts = fermi_dirac(eigval, E_fermi, kBT)
    n = counts.sum() / Hks.shape[0]

    return n

def find_E_fermi(Hks, nocc, kBT: float, E_fermi0: float=0, ntol: float=1e-8, max_iter=100.):
    E_up = E_fermi0 + 1
    E_down = E_fermi0 - 1
    nocc_converge = False

    nc = compute_nocc(Hks=Hks, E_fermi=E_fermi0, kBT=kBT)
    if abs(nc - nocc) < ntol:
        E_fermi = copy.deepcopy(E_fermi0)
        nocc_converge = True
    else:
        # first, adjust E_range
        E_range_converge = False
        while not E_range_converge:
            n_up = compute_nocc(Hks=Hks, E_fermi=E_up, kBT=kBT)
            n_down = compute_nocc(Hks=Hks, E_fermi=E_down, kBT=kBT)
            if n_up >= nocc >= n_down:
                E_range_converge = True
            else:
                if nocc > n_up:
                    E_up = E_fermi0 + (E_up - E_fermi0) * 2.
                else:
                    E_down = E_fermi0 + (E_down - E_fermi0) * 2.
        
        E_fermi = (E_up + E_down) / 2.

    # second, doing binary search  
    niter = 0

    while not nocc_converge and niter < max_iter:
        nc = compute_nocc(Hks=Hks, E_fermi=E_fermi, kBT=kBT)
        if abs(nc - nocc) < ntol:
            nocc_converge = True
        else:
            if nc < nocc:
                E_down = copy.deepcopy(E_fermi)
            else:
                E_up = copy.deepcopy(E_fermi)
            
            # update E_fermi
            E_fermi = (E_up + E_down) / 2.

        niter += 1

    if abs(nc - nocc) > ntol:
        raise RuntimeError("The Fermi Energy searching did not converge")

    return E_fermi

def fermi_dirac(energy, E_fermi, kBT):
    x = (energy - E_fermi) / kBT
    out = torch.zeros_like(x)
    out[x.lt(-500)] = 1.
    mask = x.gt(-500)*x.lt(500)
    out[mask] = 1./(torch.exp(x[mask]) + 1)
    return out

def symsqrt(matrix): # this may returns nan grade when the eigenvalue of the matrix is very degenerated.
    """Compute the square root of a positive definite matrix."""
    # _, s, v = safeSVD(matrix)
    _, s, v = torch.svd(matrix)
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
