from __future__ import annotations
from typing import Sequence, Union, Optional
from numbers import Number
import numpy as np
from . import (
    Operator,
    create_u,
    create_d,
    annihilate_u,
    annihilate_d,
    number_u,
    number_d,
    S_z,
    S_p,
    S_m,
)

def Hubbard(
    neighbors,
    nsites,
    U: Number,
    t: Union[Number, Sequence[Number]] = 1.0,
) -> Operator:
    r"""
    Hubbard Hamiltonian
    :math:`H = -t_n \sum_{<ij>_n} \sum_{s \in \{↑,↓\}} (c_{i,s}^† c_{j,s} + c_{j,s}^† c_{i,s}) + U \sum_i n_{i↑} n_{i↓}`
    """

    # index in neighbors is orbital index, rather than spin orbital index
    
    if isinstance(t, Number):
        t = [t]

    def hop(i, j):
        hop_up = create_u(nsites, i) * annihilate_u(nsites, j) + create_u(nsites, j) * annihilate_u(nsites, i)
        hop_down = create_d(nsites, i) * annihilate_d(nsites, j) + create_d(nsites, j) * annihilate_d(nsites, i)
        return hop_up + hop_down

    H = 0
    for neighbor, tn in zip(neighbors, t):
        for (i, j) in neighbor:
            H +=  -tn * hop(i, j)

    H += U * sum(number_u(nsites, i) * number_d(nsites, i) for i in range(nsites))
    return H

def Slater_Kanamori(
    nsites: int,
    t: np.ndarray,
    U: Number = 3.0,
    Up: Number = 0.0,
    J: Number = 0.0,
    Jp: Number = 0.0,
    Szpen: Number = 0.0,
    S2pen: Number = 0.0,
    n_noninteracting: Optional[int] = 0,
):
    # currently. only spin degenerate t is supported

    assert n_noninteracting <= nsites, "The number of noninteracting sites should be less than the total number of sites."

    assert t.shape[0] == 2*nsites
    assert len(t.shape) == 2, "The shape of t should be 2D."
    assert t.shape[0] == nsites*2, "t should equals to the number of spin orbitals."

    H = 0
    # we assume spin orbitals will be adjacent in the order of up, down, up, down, ...
    t = t.tolist()

    n_interacting = nsites - n_noninteracting
    
    for i in range(2*nsites):
        for j in range(2*nsites):
            if abs(t[i][j]) > 1e-7:
                if i % 2 == 0 and j % 2 == 0:
                    H += t[i][j] * create_u(nsites, i//2) * annihilate_u(nsites, j//2) # up-up
                elif i % 2 == 1 and j % 2 == 1:
                    H += t[i][j] * create_d(nsites, i//2) * annihilate_d(nsites, j//2) # down-down
                elif i % 2 == 0 and j % 2 == 1: 
                    H += t[i][j] * create_u(nsites, i//2) * annihilate_d(nsites, j//2) # up-down
                elif i % 2 == 1 and j % 2 == 0:
                    H += t[i][j] * create_d(nsites, i//2) * annihilate_u(nsites, j//2) # down-up
                else:
                    raise ValueError


    if U > 1e-7:
        H += sum(U * number_u(nsites, i) * number_d(nsites, i) for i in range(nsites-n_noninteracting))

    if Up > 1e-7:
        for i in range(nsites-n_noninteracting):
            for j in range(i+1, nsites-n_noninteracting):
                H += (Up / 2) * (number_u(nsites, i) * number_d(nsites, j) + number_d(nsites, i) * number_u(nsites, j))
                H += (Up / 2) * (number_u(nsites, i) * number_u(nsites, j) + number_d(nsites, i) * number_d(nsites, j))

                H += (Up / 2) * (number_u(nsites, j) * number_d(nsites, i) + number_d(nsites, j) * number_u(nsites, i))
                H += (Up / 2) * (number_u(nsites, j) * number_u(nsites, i) + number_d(nsites, j) * number_d(nsites, i))
    
    if J > 1e-7:
        for i in range(nsites-n_noninteracting):
            for j in range(i+1, nsites-n_noninteracting):
                H -= (J / 2) * (
                    create_u(nsites, i) * annihilate_d(nsites, i) * create_d(nsites, j) * annihilate_u(nsites, j) 
                    + create_u(nsites, j) * annihilate_d(nsites, j) * create_d(nsites, i) * annihilate_u(nsites, i)
                    )
                H -= (J / 2) * (
                    create_d(nsites, i) * annihilate_u(nsites, i) * create_u(nsites, j) * annihilate_d(nsites, j) 
                    + create_d(nsites, j) * annihilate_u(nsites, j) * create_u(nsites, i) * annihilate_d(nsites, i)
                    )

                # ############## temp #################
                # H -= (J / 2) * (number_u(nsites, i) * number_u(nsites, j) + number_u(nsites, j) * number_u(nsites, i))
                # H -= (J / 2) * (number_d(nsites, i) * number_d(nsites, j) + number_d(nsites, j) * number_d(nsites, i))
            # H += (J * (nsites-n_noninteracting-1) / 2) * (number_u(nsites, i) + number_d(nsites, i))
            # ############## temp #################

    if Jp > 1e-7:
        for i in range(nsites-n_noninteracting):
            for j in range(i+1, nsites-n_noninteracting):
                H -= (Jp / 2) * (create_u(nsites, i) * create_d(nsites, i) * annihilate_u(nsites, j) * annihilate_d(nsites, j))
                H -= (Jp / 2) * (create_u(nsites, j) * create_d(nsites, j) * annihilate_u(nsites, i) * annihilate_d(nsites, i))
    
    if Szpen > 1e-7:
        H += Szpen * sum(S_z(nsites, i) * S_z(nsites, i) for i in range(nsites))
    
    if S2pen > 1e-7:
        H += S2pen * sum(S_m(nsites, i) * S_p(nsites, i) + S_z(nsites, i) * S_z(nsites, i) + S_z(nsites, i) for i in range(nsites))
    return H # we should notice that the spinorbital are not adjacent in the quspin hamiltonian, so properties computed from this need to be transformed.
    