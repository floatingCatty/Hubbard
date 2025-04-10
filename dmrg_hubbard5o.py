from ase.io import read
from gGA.data import AtomicData
from gGA.gutz import GhostGutzwiller
from gGA.utils.tools import setup_seed
from gGA.utils.make_kpoints import kmesh_sampling
from gGA.data import block_to_feature
from gGA.utils.tools import get_semicircle_e_list
import numpy as np
from gGA.data import _keys

setup_seed(1238)
U = 1.
J = 0.25 * U # 0.25 * U
Up = U - 2*J
Jp = J
n_threads = 50

alpha = 1.
Delta = 0.25
e_list = get_semicircle_e_list(nmesh=1000)
eks = alpha * e_list[:,None,None] * np.eye(5)[None,:,:]

onsite = np.eye(5) * (-Delta)
onsite[0,0] = 0.125
onsite[1,1] = 0.125
onsite = onsite[None,:,:]
phy_onsite = {
    "C": onsite
}

intparams = {"C":[{"U":U,"Up":Up,"J":J, "Jp":Jp}]}

gga = GhostGutzwiller(
    atomic_number=np.array([6]),
    nocc=6,
    basis={"C":[5]},
    idx_intorb={"C":[0]},
    naux=3,
    intparams=intparams,
    nspin=1,
    kBT=0.0002,
    mutol=1e-4,
    natural_orbital=False,
    decouple_bath=True,
    solver="DMRG",
    mixer_options={"method": "Linear", "a": 0.5},
    iscomplex=False,
    solver_options={"reorder": True, "mpi": True, "iprint": 1, "n_threads": n_threads, "nupdate": 8, "eig_cutoff": 1e-7, "bond_dim": 1000, "su2": True}#{"mfepmin":2000, "channels": 10, "Ptol": 1e-5},
)

atomicdata = AtomicData.from_ase(
    read("./gGA/test/C_cube.vasp"),
    r_max=3.1
    )

atomicdata = AtomicData.to_AtomicDataDict(atomicdata)
atomicdata[_keys.HAMILTONIAN_KEY] = eks
atomicdata[_keys.PHY_ONSITE_KEY] = phy_onsite

gga.load("/nessa/users/zhanghao/dev/Hubbard/gGA_DMRG_H5O_B3U1D025J025_tmp.npz")
gga.run(atomicdata, 20, 5e-4, ckptprefix="DMRG_H5O_B3U1D025J025")