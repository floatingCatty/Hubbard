from ase.io import read
from gGA.data import AtomicData
from gGA.gutz import GhostGutzwiller
from gGA.utils.tools import setup_seed
from gGA.utils.make_kpoints import kmesh_sampling
from gGA.data import block_to_feature
from gGA.utils.tools import get_semicircle_e_list
import numpy as np
from gGA.data import _keys
from gGA.operator.soc import get_soc_matrix_cubic_basis

setup_seed(1238)
U = 3.
J = 0.15 * U # 0.25 * U
Up = U - 2*J
Jp = J
lbd = 0.
n_threads = 70

alpha = 1.
Delta = 0.
e_list = get_semicircle_e_list(nmesh=1000, d=2)
eks = alpha * e_list[:,None,None] * np.eye(3)[None,:,:] + 0j

onsite = np.eye(3) * (-Delta) + 0j
# SOC = get_soc_matrix_cubic_basis("d").reshape(2,5,2,5).transpose(1,0,3,2).reshape(10,10)[[0,1,2,3,6,7]][:,[0,1,2,3,6,7]]
SOC = get_soc_matrix_cubic_basis("p").reshape(2,3,2,3).transpose(1,0,3,2).reshape(6,6)
onsite = np.kron(onsite, np.eye(2)) + lbd * SOC
onsite = onsite[None,:,:]
phy_onsite = {
    "C": onsite
}

intparams = {"C":[{"U":U,"Up":Up,"J":J, "Jp":Jp}]}

gga = GhostGutzwiller(
    atomic_number=np.array([6]),
    nocc=4,
    basis={"C":[3]},
    idx_intorb={"C":[0]},
    naux=3,
    intparams=intparams,
    nspin=4,
    kBT=0.0002,
    mutol=1e-6,
    natural_orbital=False,
    decouple_bath=True,
    solver="DMRG",
    mixer_options={"method": "Linear", "a": 0.5},  # {"method": "Linear", "a":0.5},
    iscomplex=True,
    solver_options={"reorder": True, "iprint": 1, "n_threads": n_threads, "nupdate": 8, "eig_cutoff": 1e-7, "bond_dim": 3000, "su2": False}#{"mfepmin":2000, "channels": 10, "Ptol": 1e-5},
)

atomicdata = AtomicData.from_ase(
    read("./gGA/test/C_cube.vasp"),
    r_max=3.1
    )

atomicdata = AtomicData.to_AtomicDataDict(atomicdata)
atomicdata[_keys.HAMILTONIAN_KEY] = eks
atomicdata[_keys.PHY_ONSITE_KEY] = phy_onsite


gga.load("./gGA_dmrgh3o_tmp.npz")
gga.run(atomicdata, 1000, tol=1e-3, ckptprefix="dmrgh3o")