from gGA.utils.wannier import get_wannier_blocks
from gGA.utils.make_kpoints import kmesh_sampling
from gGA.gutz import GhostGutzwiller
from ase.io import read
import numpy as np
from gGA.data import AtomicData, AtomicDataDict, OrbitalMapper
from gGA.data.interfaces import block_to_feature
from gGA.gutz.hr2hk import GGAHR2HK
from gGA.utils.make_kpoints import abacus_kpath

blocks = get_wannier_blocks(
    atomic_symbol=["Eu", "O"],
    file="./gGA/test/EuO_hr.dat",
    target_basis_order={"Eu": ["d", "f"], "O": ["p"]},
    wannier_proj_orbital={"Eu": ["f", "d"], "O": ["p"]},
    orb_wan={"p": ["px", "py", "pz"], "d": ["dxy", "dyz", "dxz", "dx2-y2", "dz2"], "f":['fz3', 'fxz2', 'fyz2', 'fxyz', 'fx3-3xy2','f3x2y-y3','fx2z-y2z']}
)

# for k in blocks:
#     blocks[k] = np.kron(blocks[k], np.eye(2))


data = AtomicData.from_ase(
    atoms=read("./gGA/test/EuO.vasp"),
    r_max={"Eu": 5.5, "O": 5.5},
    pbc=True
)

atoms = data.to_ase()

idp = OrbitalMapper(basis={"Eu": "1d1f", "O":"1p"}, spin_deg=True)
hr2k = GGAHR2HK(idp_phy=idp)

kpoints, xx, _ = abacus_kpath(
    atoms,
    kpath=np.array([
        [0.0000000000,   0.0000000000,   0.0000000000,     20],
        [0.5000000000,   0.0000000000,   0.5000000000,     20],                         
        [0.6250000000,   0.2500000000,   0.6250000000,     1], 
        [0.3750000000,   0.3750000000,   0.7500000000,     20],
        [0.0000000000,   0.0000000000,   0.0000000000,     20],
        [0.5000000000,   0.5000000000,   0.5000000000,     20],
        [0.5000000000,   0.2500000000,   0.7500000000,     20],
        [0.5000000000,   0.0000000000,   0.5000000000,     1],
    ]
    )
)


block_to_feature(data=data, idp=idp, blocks=blocks)
data = AtomicData.to_AtomicDataDict(data)
data[AtomicDataDict.KPOINT_KEY] = kpoints
data = idp(data)



# setup_seed(1234)

U = 3
J = 0. # 0.25 * U
Up = U - 2*J
Jp = J

gga = GhostGutzwiller(
    atomic_number=np.array([63, 8]),
    nocc=17,
    basis={"Eu":"1d1f", "O":"1p"},
    idx_intorb={"Eu":[1]},
    naux=3,
    intparams={"Eu":[{"U":U,"Up":Up,"J":J, "Jp":Jp}]},
    nspin=1,
    kBT=0.0002,
    overlap=False,
    mutol=1e-4,
    solver="NQS",
    mixer_options={"method": "Linear", "a": 0.3},
    natural_orbital=True,
    iscomplex=True,
    solver_options={"mfepmax":2, "nnepmax":10000, "channels": 10, "Ptol": 1e-3, "Etol": 1e-3, "mftol": 1e-2, "Nsamples": 1000},
)

data["kpoint"] = kmesh_sampling([10,10,10], True)
gga.run(data, 200, 1e-3)