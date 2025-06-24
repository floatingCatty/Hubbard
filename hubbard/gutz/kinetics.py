from hubbard.data import OrbitalMapper, AtomicDataDict, _keys
from typing import Tuple, Union, Dict
from hubbard.gutz.hr2hk import GGAHR2HK
from hubbard.utils.tools import float2comlex
from hubbard.utils.constants import atomic_num_dict_r
import math
import numpy as np
import copy
from scipy.linalg import block_diag

class Kinetic(object):
    def __init__(
            self,
            nocc: int,
            naux: int,
            idp_phy: Union[OrbitalMapper, None]=None, 
            basis: Dict[str, Union[str, list]]=None,
            idx_intorb: Dict[str, list]=None,
            kBT: float = 0.001, # 0.0257,
            delta_deg=1e-4,
            mutol=1e-4,
            overlap: bool = False,
            soc: int = False, # control whether spin-orbital coupling is included, if it is true, nspin will be set as 4.
            nspin: int = 1, # control the spin orbitals, 1 for spin degenerate, 2 for spin collinear polarization, 4 for spin non-collinear polarization
            iscomplex=False,
            dtype=np.float64
            ):
        
        super(Kinetic, self).__init__()
        self.ctype = float2comlex(dtype=dtype)
        self.nocc = nocc
        self.kBT = kBT
        self.idx_intorb = idx_intorb
        self.naux = naux
        self.soc = soc
        self.spin_deg = nspin <= 1
        self.nspin = nspin
        self.iscomplex = iscomplex
        self.dtype = dtype
        self.mutol = mutol
        self.overlap = overlap

        if soc:
            self.nspin = 4
        else:
            self.nspin = nspin
        
        # assert overlap == False, "The overlap is not implemented yet."

        if basis is not None:
            self.idp_phy = OrbitalMapper(basis, method="e3tb", spin_deg=self.spin_deg)
            if idp_phy is not None:
                assert idp_phy == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp_phy is not None, "Either basis or idp should be provided."
            assert idp_phy.method == "e3tb", "The method of idp should be e3tb."
            self.idp_phy = idp_phy
            if spin_deg is not None:
                assert spin_deg == idp_phy.spin_deg, "The spin_deg of idp and spin_deg should be the same."

        nauxorbs = copy.deepcopy(self.idp_phy.listnorbs) # {'C': [1, 3], 'Si': [1, 1, 3, 3, 5]}
        for sym in self.idx_intorb.keys():
            for i in self.idx_intorb[sym]:
                nauxorbs[sym][i] = nauxorbs[sym][i] * naux
        self.nauxorbs = nauxorbs

        self.sym_to_norbphy = {}
        self.sym_to_norb = {}

        for sym in self.nauxorbs:
            self.sym_to_norb[sym] = sum(self.nauxorbs[sym])
            self.sym_to_norbphy[sym] = sum(self.idp_phy.listnorbs[sym])
        
        self.hr2hk = GGAHR2HK(
            idp_phy=self.idp_phy,
            overlap=False,
            spin_deg=self.spin_deg,
            dtype=dtype,
            idx_intorb=idx_intorb,
        )

        if self.overlap:
            self.sr2sk = GGAHR2HK(
                idp_phy=self.idp_phy,
                overlap=True,
                spin_deg=self.spin_deg,
                dtype=dtype,
                idx_intorb=idx_intorb,
                edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                node_field=AtomicDataDict.NODE_OVERLAP_KEY
            )

        self.basis = self.idp_phy.basis
        self.delta_deg = delta_deg

        start_int_orbs = {sym:[] for sym in self.idx_intorb.keys()}
        start_phy_orbs = {sym:[] for sym in self.idx_intorb.keys()}
        for sym in self.idx_intorb.keys():
            for i in self.idx_intorb[sym]:
                start_int_orbs[sym].append(sum(nauxorbs[sym][:i])*2)
                start_phy_orbs[sym].append(sum(self.idp_phy.listnorbs[sym][:i]))
        self.start_int_orbs = start_int_orbs
        self.start_phy_orbs = start_phy_orbs


    def update(self, data: AtomicDataDict.Type, R: Dict[str, np.ndarray], LAM: Dict[str, np.ndarray]) -> AtomicDataDict.Type:
        if data.get(_keys.ATOMIC_NUMBERS_KEY) is not None:
            atomic_number = data[AtomicDataDict.ATOMIC_NUMBERS_KEY].flatten().copy()
            data = self.idp_phy(data)
            atom_type = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        else:
            assert data.get(AtomicDataDict.ATOM_TYPE_KEY) is not None, "The atomic number or atom type should be provided."
            atom_type = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().copy()
            atomic_number = self.idp_phy.untransform_atom(atom_type)

        # phy_onsite, real_C2, tkR_C2, E_fermi = self._compute_H(data, R, LAM)
        phy_onsite, H, tkR, S = self._compute_H(data, R, LAM)
        real_C2, tkR_C2, E_fermi = self._compute_RDM(H=H, tkR=tkR, S=S)

        # solve for D
        B = []
        A = []
        ist = 0
        ist_phy = 0
        out_RDM = {sym: [] for sym in self.idp_phy.type_names}
        for i, an in enumerate(atomic_number):
            sym = atomic_num_dict_r[int(an)]
            norb = self.sym_to_norbphy[sym] * 2
            nauxorb = self.sym_to_norb[sym] * 2
            iatkRC2 = tkR_C2[ist_phy:ist_phy+norb, ist:ist+nauxorb]
            iaC2 = real_C2[ist:ist+nauxorb, ist:ist+nauxorb]
            sel_iaC2 = [np.eye(j*2) * 0.9 for j in self.idp_phy.listnorbs[sym]]
            sel_iatkRC2 = [np.eye(j*2) for j in self.idp_phy.listnorbs[sym]]

            if self.idx_intorb.get(sym) is not None:
                for jx, j in enumerate(self.idx_intorb[sym]):
                    sel_iaC2[j] = iaC2[self.start_int_orbs[sym][jx]:self.start_int_orbs[sym][jx]+self.nauxorbs[sym][j]*2, 
                                        self.start_int_orbs[sym][jx]:self.start_int_orbs[sym][jx]+self.nauxorbs[sym][j]*2]
                    sel_iatkRC2[j] = iatkRC2[self.start_phy_orbs[sym][jx]:self.start_phy_orbs[sym][jx]+self.idp_phy.listnorbs[sym][j]*2,
                                        self.start_int_orbs[sym][jx]:self.start_int_orbs[sym][jx]+self.nauxorbs[sym][j]*2]
                A.append(block_diag(*sel_iaC2))
                B.append(block_diag(*sel_iatkRC2))

            out_RDM[sym].append(iaC2)

            ist += nauxorb
            ist_phy += norb

        A = block_diag(*A)
        B = block_diag(*B).T

        # print(torch.linalg.eigvalsh(A @ (torch.eye(A.shape[0], device=self.device) - A)))
        A = symsqrt(A @ (np.eye(A.shape[0]) - A))
        # print(torch.linalg.eigvalsh(A))
        try:
            print("DM_kin: ", np.linalg.eigvalsh(real_C2))
            D = np.linalg.solve(A, B)
        except:
            import matplotlib.pyplot as plt
            plt.matshow(real_C2, cmap="bwr", vmax=1.0, vmin=-1.0)
            plt.show()
            raise RuntimeError("The solve does not converge.")
        
        out_D = {sym: [] for sym in self.idp_phy.type_names}

        # split D and RDM
        ist = 0
        ist_phy = 0
        for i, an in enumerate(atomic_number):
            sym = atomic_num_dict_r[int(an)]
            norb = self.sym_to_norb[sym] * 2
            norb_phy = self.sym_to_norbphy[sym] * 2
            if self.idx_intorb.get(sym) is not None:
                iaD = D[ist:ist+norb, ist_phy:ist_phy+norb_phy]
                sel_iaD = [np.eye(j*2) for j in self.idp_phy.listnorbs[sym]]
                for jx, j in enumerate(self.idx_intorb[sym]):
                    sel_iaD[j] = iaD[self.start_int_orbs[sym][jx]:self.start_int_orbs[sym][jx]+self.nauxorbs[sym][j]*2, 
                                    self.start_phy_orbs[sym][jx]:self.start_phy_orbs[sym][jx]+self.idp_phy.listnorbs[sym][j]*2]
                
                out_D[sym].append(block_diag(*sel_iaD))

                ist += norb
                ist_phy += norb_phy
        
        for sym in self.idp_phy.type_names:
            if sym in self.idx_intorb.keys():
                out_D[sym] = np.stack(out_D[sym])
            out_RDM[sym] = np.stack(out_RDM[sym])

        return phy_onsite, out_D, out_RDM, E_fermi

    def _compute_H(self, data: AtomicDataDict.Type, R: Dict[str, np.ndarray], LAM: Dict[str, np.ndarray]):

        # construct LAM
        if data.get(AtomicDataDict.HAMILTONIAN_KEY) is None:
            kpoints = data[AtomicDataDict.KPOINT_KEY]
            phy_onsite, H, tkR = self.hr2hk(data, R, LAM)
            if self.overlap: # we assume here the interacting orbitals are orthogonal, which is true for LCAO or Wannier basis.
                _, S, _ = self.sr2sk(data, R)
            else:
                S = None
        else:
            hamil = data[AtomicDataDict.HAMILTONIAN_KEY]
            if self.overlap:
                ovp = data[AtomicDataDict.OVERLAP_KEY]
            phy_onsite = data[AtomicDataDict.PHY_ONSITE_KEY]
            # assert hamil.shape[0] == kpoints.shape[0]

            Rs = []
            LAMs = []
            type_count = [0] * len(self.idp_phy.type_names)
            for i, at in enumerate(data[AtomicDataDict.ATOM_TYPE_KEY].flatten()):
                idx = type_count[at]
                sym = self.idp_phy.type_names[at]
                if R is not None:
                    if sym in self.idx_intorb.keys():
                        Rs.append(R[sym][idx].conj().T)
                    else:
                        Rs.append(np.eye(sum(self.nauxorbs[sym])*2).astype(self.dtype))
                
                if LAM is not None:
                    if sym in self.idx_intorb.keys():
                        LAMs.append(LAM[sym][idx])
                    else:
                        n = sum(self.nauxorbs[sym])*2
                        LAMs.append(np.zeros((n,n)).astype(self.dtype))

                type_count[at] += 1

            # transfer orbitals to spinorbitals

            Rs = block_diag(*Rs)[None,...].astype(hamil.dtype)
            LAMs = block_diag(*LAMs)[None,...].astype(hamil.dtype)

            nphyspin = Rs.shape[1]

            # check spin deg of hamil
            if hamil.shape[1] == nphyspin // 2:
                hamil = np.stack([hamil, np.zeros_like(hamil), np.zeros_like(hamil), hamil]).reshape(2,2,-1,nphyspin//2,nphyspin//2).transpose(2,3,0,4,1).reshape(-1,nphyspin,nphyspin)
            else:
                assert hamil.shape[1] == hamil.shape[2] == nphyspin
            
            if self.overlap:
                if ovp.shape[1] == nphyspin // 2:
                    ovp = np.stack([ovp, np.zeros_like(ovp), np.zeros_like(ovp), ovp]).reshape(2,2,-1,nphyspin//2,nphyspin//2).transpose(2,3,0,4,1).reshape(-1,nphyspin,nphyspin)
                else:
                    assert ovp.shape[1] == ovp.shape[2] == nphyspin

            # check spin deg of phyonsite
            for sym in phy_onsite:
                shape = phy_onsite[sym].shape
                nsym_auxspin = sum(self.idp_phy.listnorbs[sym])*2
                if shape[1] == nsym_auxspin // 2: # spin deg
                    phy_onsite[sym] = np.stack([phy_onsite[sym], np.zeros_like(phy_onsite[sym]), np.zeros_like(phy_onsite[sym]), phy_onsite[sym]]).reshape(2,2,-1,nsym_auxspin//2,nsym_auxspin//2).transpose(2,3,0,4,1).reshape(-1,nsym_auxspin,nsym_auxspin)
                else:
                    assert shape[1] == shape[2] == nsym_auxspin
                    
            tkR = hamil @ Rs
            H = Rs.conj().transpose(0,2,1) @ tkR + LAMs
            if self.overlap:
                S = Rs.conj().transpose(0,2,1) @ ovp @ Rs
            else:
                S = None

        return phy_onsite, H, tkR, S
    
    def _compute_RDM(self, H, S, tkR, E_fermi: float=None):

        nk = H.shape[0]
        if self.overlap:
            chklowt = np.linalg.cholesky(S)
            chklowtinv = np.linalg.inv(chklowt)
            H = chklowtinv @ H @ np.transpose(chklowtinv,(0,2,1)).conj()

        eigval, eigvec = np.linalg.eigh(H)

        norb = eigval.shape[1]
        eigvec = eigvec.reshape(nk, -1, norb).transpose(0,2,1)
        
        if E_fermi is None:
            E_fermi = np.sort(eigval.flatten())[math.ceil(self.nocc)*nk-1:math.ceil(self.nocc)*nk+1].sum() * 0.5
            E_fermi = find_E_fermi(Hks=H, nocc=self.nocc, kBT=self.kBT, E_fermi0=E_fermi, ntol=self.mutol)

        factor = fermi_dirac(eigval, E_fermi, self.kBT) # [nk, norb]
        mask = factor > 1e-10 # [nk, norb]
        scatter_index = np.arange(nk).reshape(-1,1).repeat(norb, axis=1).reshape(-1) # [nk*norb]
        scatter_index = scatter_index[mask.flatten()]
        vec_m = (eigvec * factor[...,None])[mask]
        real_C2_ = np.einsum("ni, nj->nij", vec_m, eigvec[mask].conj()) # nstates, norb*2, norb*2
        real_C2 = np.zeros([nk]+list(real_C2_.shape[1:]), dtype=real_C2_.dtype)
        np.add.at(real_C2, scatter_index, real_C2_)
        real_C2 = real_C2.swapaxes(1,2)
        # real_C2 = scatter(real_C2, scatter_index, dim=0, reduce="sum", dim_size=nk).transpose(1,2)

        tkR_C2 = (tkR @ real_C2.transpose(0,2,1)).sum(0) / nk
        # check imaginary part
        if not self.iscomplex:
            assert np.abs(tkR_C2.imag).max() < 1e-8, "The imaginary part of tkR_C2 {} is not zero.".format(np.abs(tkR_C2.imag).max())
            assert np.abs(real_C2.imag).max() < 1e-8, "The imaginary part of real_C2 {} is not zero.".format(np.abs(real_C2.imag).max())

            tkR_C2 = tkR_C2.real
            real_C2 = real_C2.real.sum(0) / nk
        else:
            real_C2 = real_C2.sum(0) / nk

        return real_C2, tkR_C2, E_fermi


def compute_nocc(Hks, E_fermi: float, kBT: float):
    """
        Hks: [nk, norb, norb],
    """
    eigval = np.linalg.eigvalsh(Hks)
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
    out = np.zeros_like(x)
    out[x < -500] = 1.
    mask = (x > -500) * (x < 500)
    out[mask] = 1./(np.exp(x[mask]) + 1)
    return out

def symsqrt(matrix): # this may returns nan grade when the eigenvalue of the matrix is very degenerated.
    """Compute the square root of a positive definite matrix."""
    # _, s, v = safeSVD(matrix)
    _, s, v = np.linalg.svd(matrix)
    v = v.conj().T

    good = s > s.max(axis=-1, keepdims=True) * s.shape[-1] * np.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.shape[-1]:
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, np.zeros((), dtype=s.dtype))
    
    shape = list(s.shape[:-1]) + [1] + [s.shape[-1]]
    return (v * np.sqrt(s).reshape(shape)) @ v.conj().swapaxes(-1,-2)
