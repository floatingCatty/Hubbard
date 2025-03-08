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

alpha = 1.
Delta = 0.25
e_list = get_semicircle_e_list(nmesh=1000)
eks = alpha * e_list[:,None,None] * np.eye(5)[None,:,:]

onsite = np.eye(5) * Delta
onsite[0,0] = 0.
onsite[1,1] = 0.
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
    nspin=4,
    kBT=0.0002,
    mutol=1e-4,
    natural_orbital=False,
    decouple_bath=True,
    solver="DMRG",
    mixer_options={"method": "Linear", "a": 0.3},
    iscomplex=False,
    solver_options={"n_threads": 50, "nupdate": 8, "eig_cutoff": 1e-7, "bond_dim": 2500}#{"mfepmin":2000, "channels": 10, "Ptol": 1e-5},
)


atomicdata = AtomicData.from_ase(
    read("./gGA/test/C_cube.vasp"),
    r_max=3.1
    )

atomicdata = AtomicData.to_AtomicDataDict(atomicdata)
atomicdata[_keys.HAMILTONIAN_KEY] = eks
atomicdata[_keys.PHY_ONSITE_KEY] = phy_onsite


gga.run(atomicdata, 500, 5e-4)
gga.save(f="./gGA/test/H5o", prefix="DMRG_B3U1D025J025")