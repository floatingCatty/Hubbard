import numpy as np
from hubbard.operator.common_operators import (
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

from .lattice import (
    LatticeBond,
    nnn_square_lattice,
    iter_neighbors_and_types,
)


def _single_term(opstr: str, coeff: float, *indices: int) -> Operator:
    return Operator([[opstr, [[float(coeff), *map(int, indices)]]]])


def _simplify_quartic_term(
    coeff: float,
    p: int,
    q: int,
    r: int,
    s: int,
) -> Operator:
    """Return a canonical form for c_p^dagger c_q^dagger c_r c_s.

    EH generates many quartic terms with repeated indices. Leaving them as a
    literal ``++--`` string makes Quantax classify them as four-flip updates even
    when their action is diagonal or only changes two occupations. We reduce them
    here so the operator string reflects the true update rank seen by the sampler.
    """
    if p == q or r == s:
        return 0

    if q == r:
        if p == s:
            return _single_term("nn", coeff, p, q)
        return _single_term("+n-", coeff, p, q, s)

    if p == s:
        return _single_term("n+-", coeff, p, q, r)

    if q == s:
        if p == r:
            return _single_term("nn", -coeff, p, q)
        return _single_term("+n-", -coeff, p, q, r)

    if p == r:
        return _single_term("n+-", -coeff, p, q, s)

    return _single_term("++--", coeff, p, q, r, s)


def _spin_orbital_index(nsites: int, orbital: int, spin: str) -> int:
    if spin == "u":
        return orbital
    if spin == "d":
        return orbital + nsites
    raise ValueError(f"Unknown spin label {spin!r}")

def multi_orbital_extended_hubbard(
    Nx: int,
    Ny: int,
    chem_1: float,
    chem_2: float,
    **kwargs
) -> Operator:
    """
    Two-orbital (1s-A1 + 2s-A1) extended Hubbard model on a square lattice,
    translated from the Julia code.
    """

    # Lattice size
    N = Nx * Ny
    nsites = 2 * N

    # Tunnel coupling strengths
    t_nn = 1.0
    t_nnn = 0.5
    t_nn_2s = 1.5
    t_nnn_2s = 0.75

    # Onsite energies
    epsilon = -74.0
    epsilon_2s = epsilon + (43.59 - 14.79)

    # Inter-site Coulomb interaction strengths
    w1 = 18.0
    w2 = 12.0

    # Lattice creation 
    lattice_a1 = nnn_square_lattice(Nx, Ny, yperiodic=False)

    # Coulomb matrix
    coulomb_matrix = np.array([[ 9.05207631e-21-0.j, -4.14272356e-24-0.j, -1.91691240e-21-0.j,
                                -4.14272356e-24-0.j,  9.40678813e-24-0.j,  2.07849430e-25-0.j,
                                -1.91691240e-21-0.j,  2.07849430e-25-0.j,  1.04039318e-21-0.j],
                                [-4.14272356e-24+0.j,  6.95137797e-21-0.j,  8.03638974e-23-0.j,
                                9.40678813e-24-0.j, -3.14663683e-24-0.j, -1.75336598e-23-0.j,
                                2.07849423e-25-0.j, -7.96530880e-22-0.j,  8.87592050e-24-0.j],
                                [-1.91691240e-21+0.j,  8.03638974e-23+0.j,  6.98201024e-21-0.j,
                                2.07849423e-25+0.j, -1.75336598e-23-0.j, -1.13912439e-25-0.j,
                                1.04039318e-21-0.j,  8.87592050e-24-0.j, -8.80100343e-22-0.j],
                                [-4.14272356e-24+0.j,  9.40678813e-24+0.j,  2.07849423e-25-0.j,
                                6.95137797e-21-0.j, -3.14663683e-24-0.j, -7.96530880e-22-0.j,
                                8.03638974e-23-0.j, -1.75336598e-23-0.j,  8.87592050e-24-0.j],
                                [ 9.40678813e-24+0.j, -3.14663683e-24+0.j, -1.75336598e-23+0.j,
                                -3.14663683e-24+0.j,  6.07935945e-21-0.j,  8.68230737e-23-0.j,
                                -1.75336598e-23+0.j,  8.68230737e-23-0.j,  7.17192481e-23-0.j],
                                [ 2.07849430e-25+0.j, -1.75336598e-23+0.j, -1.13912439e-25+0.j,
                                -7.96530880e-22+0.j,  8.68230737e-23+0.j,  5.98100687e-21-0.j,
                                8.87592050e-24+0.j,  7.17192481e-23-0.j,  7.04978522e-23-0.j],
                                [-1.91691240e-21+0.j,  2.07849423e-25+0.j,  1.04039318e-21+0.j,
                                8.03638974e-23+0.j, -1.75336598e-23-0.j,  8.87592050e-24-0.j,
                                6.98201024e-21-0.j, -1.13912439e-25-0.j, -8.80100343e-22-0.j],
                                [ 2.07849430e-25+0.j, -7.96530880e-22+0.j,  8.87592050e-24+0.j,
                                -1.75336598e-23+0.j,  8.68230737e-23+0.j,  7.17192481e-23+0.j,
                                -1.13912439e-25+0.j,  5.98100687e-21-0.j,  7.04978522e-23-0.j],
                                [ 1.04039318e-21+0.j,  8.87592050e-24+0.j, -8.80100343e-22+0.j,
                                8.87592050e-24+0.j,  7.17192481e-23+0.j,  7.04978522e-23+0.j,
                                -8.80100343e-22+0.j,  7.04978522e-23+0.j,  6.00901751e-21-0.j]]).reshape(3,3,3,3)
    coulomb_matrix_mev = coulomb_matrix / 1.60218e-19 * 1000.0  # J to meV

    inds = np.array([0, 2], dtype=int)
    colmat = coulomb_matrix_mev[np.ix_(inds, inds, inds, inds)]

    H = 0

    # Hopping on 1s-A1 orbital
    for (i, j), bond_type in iter_neighbors_and_types(lattice_a1):
        if bond_type == "nn":
            t = t_nn
        else:
            t = t_nnn

        hop_up = (
            create_u(nsites, i) * annihilate_u(nsites, j)
            + create_u(nsites, j) * annihilate_u(nsites, i)
        )
        hop_dn = (
            create_d(nsites, i) * annihilate_d(nsites, j)
            + create_d(nsites, j) * annihilate_d(nsites, i)
        )
        H += -t * (hop_up + hop_dn)

    # Onsite energies for 1s-A1 orbital
    for j in range(N):
        H += epsilon * (number_u(nsites, j) + number_d(nsites, j))

    # Hopping on 2s-A1 orbital
    for (i, j), bond_type in iter_neighbors_and_types(lattice_a1):
        i2 = i + N
        j2 = j + N

        if bond_type == "nn":
            t2 = t_nn_2s
        else:
            t2 = t_nnn_2s

        hop_up = (
            create_u(nsites, i2) * annihilate_u(nsites, j2)
            + create_u(nsites, j2) * annihilate_u(nsites, i2)
        )
        hop_dn = (
            create_d(nsites, i2) * annihilate_d(nsites, j2)
            + create_d(nsites, j2) * annihilate_d(nsites, i2)
        )
        H += -t2 * (hop_up + hop_dn)

    # onsite energy for 2s-A1 orbital
    for j in range(N, 2 * N):
        H += epsilon_2s * (number_u(nsites, j) + number_d(nsites, j))

    # On-site Coulomb interactions
    for site in range(N):
        for i_orb in (0, 1):
            for j_orb in (0, 1):
                for k_orb in (0, 1):
                    for l_orb in (0, 1):
                        i_site = site + (0 if i_orb == 0 else N)
                        j_site = site + (0 if j_orb == 0 else N)
                        k_site = site + (0 if k_orb == 0 else N)
                        l_site = site + (0 if l_orb == 0 else N)

                        V = float(0.5 * colmat[i_orb, j_orb, l_orb, k_orb].real)
                        if abs(V) < 1e-5:
                            continue
                        
                        
                        H += _simplify_quartic_term(
                            V,
                            _spin_orbital_index(nsites, i_site, "u"),
                            _spin_orbital_index(nsites, j_site, "u"),
                            _spin_orbital_index(nsites, k_site, "u"),
                            _spin_orbital_index(nsites, l_site, "u"),
                        )
                        H += _simplify_quartic_term(
                            V,
                            _spin_orbital_index(nsites, i_site, "u"),
                            _spin_orbital_index(nsites, j_site, "d"),
                            _spin_orbital_index(nsites, k_site, "d"),
                            _spin_orbital_index(nsites, l_site, "u"),
                        )
                        H += _simplify_quartic_term(
                            V,
                            _spin_orbital_index(nsites, i_site, "d"),
                            _spin_orbital_index(nsites, j_site, "u"),
                            _spin_orbital_index(nsites, k_site, "u"),
                            _spin_orbital_index(nsites, l_site, "d"),
                        )
                        H += _simplify_quartic_term(
                            V,
                            _spin_orbital_index(nsites, i_site, "d"),
                            _spin_orbital_index(nsites, j_site, "d"),
                            _spin_orbital_index(nsites, k_site, "d"),
                            _spin_orbital_index(nsites, l_site, "d"),
                        )


    # Inter-site Coulomb interactions
    for (i, j), bond_type in iter_neighbors_and_types(lattice_a1):
        W = w1 if bond_type == "nn" else w2

        for m in (0, 1): 
            for n in (0, 1):
                s_i = i + m * N
                s_j = j + n * N

                H += W * (
                    number_u(nsites, s_i) * number_u(nsites, s_j)
                    + number_u(nsites, s_i) * number_d(nsites, s_j)
                    + number_d(nsites, s_i) * number_u(nsites, s_j)
                    + number_d(nsites, s_i) * number_d(nsites, s_j)
                )

    # Gate-induced chemical potentials change on two chosen sites
    x_1, y_1 = 1, 1
    x_2, y_2 = 2, 1

    j_1 = (x_1 - 1) * Ny + (y_1 - 1)
    j_2 = (x_2 - 1) * Ny + (y_2 - 1)

    H += chem_1 * (number_u(nsites, j_1) + number_d(nsites, j_1))
    H += chem_2 * (number_u(nsites, j_2) + number_d(nsites, j_2))

    H += chem_1 * (number_u(nsites, j_1 + N) + number_d(nsites, j_1 + N))
    H += chem_2 * (number_u(nsites, j_2 + N) + number_d(nsites, j_2 + N))

    return H
