from hubbard.utils.wannier import get_wannier_blocks
from ase.io import read
import numpy as np
from hubbard.data import AtomicData, AtomicDataDict, OrbitalMapper
from hubbard.data.interfaces import block_to_feature
from hubbard.gutz.hr2hk import GGAHR2HK
from hubbard.utils.make_kpoints import abacus_kpath, kmesh_sampling
from hubbard.gutz import GhostGutzwiller
import numpy as np

blocks = get_wannier_blocks(
    atomic_symbol=["Eu", "O"],
    file="./gGA/test/EuO_hr.dat",
    target_basis_order={"Eu": ["d", "f"], "O": ["p"]},
    wannier_proj_orbital={"Eu": ["f", "d"], "O": ["p"]},
    orb_wan={"p": ["px", "py", "pz"], "d": ["dxy", "dyz", "dxz", "dx2-y2", "dz2"], "f":['fz3', 'fxz2', 'fyz2', 'fxyz', 'fx3-3xy2','f3x2y-y3','fx2z-y2z']},
    spinors=True
)

idp = OrbitalMapper(basis={"Eu": "1d1f", "O":"1p"}, spin_deg=False)

data = AtomicData.from_ase(
    atoms=read("./gGA/test/EuO.vasp"),
    r_max={"Eu": 12., "O": 12.},
    pbc=True
)
block_to_feature(data=data, idp=idp, blocks=blocks)
data = AtomicData.to_AtomicDataDict(data)
# data = data.to_AtomicDataDict(data)
# %env OMP_NUM_THREADS=20

# setup_seed(1234)

U = 7.1
J = 0. # 0.25 * U
Up = U - 2*J
Jp = J

gga = GhostGutzwiller(
    atomic_number=np.array([63, 8]),
    nocc=13,
    basis={"Eu":"1d1f", "O":"1p"},
    idx_intorb={"Eu":[1]},
    naux=1,
    intparams={"Eu":[{"U":U,"Up":Up,"J":J, "Jp":Jp}]},
    nspin=4,
    kBT=0.0002,
    mutol=1e-4,
    solver="DMRG",
    decouple_bath=False,
    mixer_options={"method": "Linear", "a": 0.6},
    iscomplex=True,
    solver_options={"n_threads": 40, "nupdate":8, "bond_dim":1500} # {"mfepmin":2000, "channels": 10, "Ptol": 1e-3},
)

data["kpoint"] = kmesh_sampling([8,8,8], True)

gga.run(data, 200, 1e-3)

gga.save(f="./", prefix="dmrgEuO")