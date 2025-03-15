from ase.io import read
from gGA.data import AtomicData
from gGA.gutz import GhostGutzwiller
from gGA.utils.tools import setup_seed
from gGA.utils.make_kpoints import kmesh_sampling
from gGA.data import block_to_feature
from gGA.utils.tools import get_semicircle_e_list
import numpy as np
from gGA.data import _keys
import jax
import os
import argparse

parser = argparse.ArgumentParser(description='NQS solver for gDMET calculation on d band Hubbard model.')
parser.add_argument('--coord_addr', type=str, default="127.0.0.1:34567",help='')

def main():
    args = parser.parse_args()

    world_size = int(os.environ["SLURM_NTASKS"])
    rank = int(os.environ["SLURM_PROCID"])

    # Here we let jax know there's more than one node in this job
    jax.distributed.initialize(coordinator_address=args.coord_addr, num_processes=world_size, process_id=rank)

    print("avail devices: ", jax.devices(), "for process: ", jax.process_index())

    setup_seed(12389)
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
        nspin=1,
        kBT=0.0002,
        mutol=1e-6,
        natural_orbital=False,
        decouple_bath=True,
        solver="NQS",
        mixer_options={"method": "PDIIS", "a": 0.3},
        iscomplex=False,
            solver_options={
            "mfepmax":800,
            "nnepmax":1000,
            "d_emb": 20,
            "Nsamples": 2000,
            "nblocks":3,
            "hidden_channels":16,
            "out_channels":16,
            "ffn_hidden":[16],
            "heads":4,
            "Ptol": 5e-4, # this value should be smaller than at least 5e-4 if expecting 1e-4 convergence
            "Etol": 1e-3
            },
    )


    atomicdata = AtomicData.from_ase(
        read("./gGA/test/C_cube.vasp"),
        r_max=3.1
        )

    atomicdata = AtomicData.to_AtomicDataDict(atomicdata)
    atomicdata[_keys.HAMILTONIAN_KEY] = eks
    atomicdata[_keys.PHY_ONSITE_KEY] = phy_onsite


    gga.run(atomicdata, 10, 5e-4)
    gga.save(f="./gGA/test/H5o", prefix="NQS_B3U1D025J025")

if __name__ == "__main__":
    main()