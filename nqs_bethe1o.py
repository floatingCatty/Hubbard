from ase.io import read
from gGA.data import AtomicData
from gGA.gutz import GhostGutzwiller
from gGA.utils.tools import setup_seed
from gGA.utils.tools import get_semicircle_e_list
import numpy as np
from gGA.data import _keys

setup_seed(123)
U = 0.5
J = 0.0 * U # 0.25 * U
Up = U - 2*J
Jp = J

alpha = 1.
V = 1.0
ep = -1.0
e_list = get_semicircle_e_list(nmesh=1000)

eks = np.zeros((len(e_list), 2, 2))+0j
eks += np.array([[[0.,  V],
                 [V,   ep]]],dtype=np.complex128)
eks[:,1,1] += alpha * e_list

phy_onsite = {
    "C": np.array([
        [[-U/2, 0.],
        [0., 0.]]
    ]),
}

intparams = {"C":[{"U":U,"Up":Up,"J":J, "Jp":Jp}]}

gga = GhostGutzwiller(
    atomic_number=np.array([6]),
    nocc=3,
    basis={"C":[1,1]},
    idx_intorb={"C":[0]},
    naux=3,
    intparams=intparams,
    nspin=2,
    kBT=0.0002,
    mutol=1e-6,
    solver="NQS",
    mixer_options={"method": "Linear", "a": 0.3},
    iscomplex=False,
    solver_options={
        "mfepmax":500,
        "nnepmax":5000,
        "d_emb": 4,
        "nblocks":3,
        "hidden_channels":8,
        "out_channels":8,
        "ffn_hidden":[8],
        "heads":4,
        "Ptol": 1e-4, # this value should be smaller than at least 2e-4 if expecting 1e-3 convergence
        "Etol": 6e-4
        },
)

atomicdata = AtomicData.from_ase(
    read("./gGA/test/C_cube.vasp"),
    r_max=3.1
    )

# atomicdata["kpoint"] = torch.tensor(kmesh_sampling([10,10,10], True)).to(torch.get_default_dtype())
# block_to_feature(atomicdata, gga.kinetic.idp_phy, block)
atomicdata = AtomicData.to_AtomicDataDict(atomicdata)
atomicdata[_keys.HAMILTONIAN_KEY] = eks
atomicdata[_keys.PHY_ONSITE_KEY] = phy_onsite

# gga.load("./gGA/test/nqs/bethe1o/stateU5P-1P1e-4E1e-3.npz")
gga.run(atomicdata, 500, 1e-3)
gga.save("./gGA/test/nqs/bethe1o", "U05P-1P1e-4E6e-4")