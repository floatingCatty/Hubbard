import numpy as np
import copy

def hartree_fock(
    h_mat,
    n_imp,
    n_bath,
    nocc,
    U,
    Up,
    J,
    Jp,
    max_iter=500,
    tol=1e-6,
    kBT=1e-5,
    ntol=1e-4,
    mixing=0.3,
    verbose=True
):
    """
    Perform a Generalized Hartree-Fock (GHF) calculation for an impurity+bath system 
    with Slater-Kanamori interactions on the impurity orbitals.

    Parameters
    ----------
    h_mat : ndarray, shape (2*(n_imp + n_bath), 2*(n_imp + n_bath))
        One-body Hamiltonian (includes spin indices).
    n_imp : int
        Number of impurity orbitals (not counting spin). 
        Total spin-orbitals for impurity = 2 * n_imp.
    n_bath : int
        Number of bath orbitals (not counting spin).
        Total spin-orbitals for bath = 2 * n_bath.
    U : float
        Intra-orbital Coulomb interaction, U * n_{m↑} n_{m↓} on each impurity orbital m.
    Up : float
        Inter-orbital Coulomb interaction.
    J : float
        Hund's rule coupling (exchange).
    Jp : float
        Pair-hopping term.
    nocc : int
        Number of electrons to occupy in the single-particle space.
    max_iter : int, optional
        Maximum SCF iterations.
    tol : float, optional
        Convergence tolerance on the density matrix difference (Frobenius norm).
    mixing : float, optional
        Mixing parameter for density matrix updates (0 < mixing <=1).
    verbose : bool, optional
        If True, prints iteration details.

    Returns
    -------
    F_total : ndarray
        Final Fock matrix (normal part).
    Delta : ndarray
        Final anomalous Fock matrix (pairing part).
    D : ndarray
        Final normal density matrix.
    P : ndarray
        Final anomalous density matrix.
    eigvals : ndarray
        Final eigenvalues of the generalized Fock matrix.
    scf_history : list
        List of tuples (iteration, energy, rms_density_diff).
    """
    # Total number of orbitals (impurity + bath) *per spin*:
    n_orb = n_imp + n_bath
    # Total dimension (spin up + spin down):
    dim = 2 * n_orb

    # Helper functions

    # Initialize density matrices
    D = np.eye(dim) * (nocc / dim)  # Initial guess: uniform occupancy

    # scf_history = []
    E_prev = 0.0
    efermi = 0.

    for iteration in range(max_iter):
        # Build mean-field potentials
        F = build_mean_field(n_imp=n_imp, n_bath=n_bath, h_mat=h_mat, D=D, U=U, J=J, Up=Up)

        # Compute new density matrices
        D_new, efermi = compute_density_matrices(F=F, nocc=nocc, norb=n_orb, kBT=kBT, efermi0=efermi, ntol=ntol)

        # Compute energy
        E_tot = compute_energy(h_mat, F, D_new)

        # Compute density difference
        d = np.linalg.norm(D_new - D)

        # scf_history.append((iteration, E_tot, d))

        if verbose:
            print(f"Iter {iteration:3d}: E = {E_tot:.6f}, ΔD = {d:.6e}")

        # Check convergence
        if d < tol:
            D = D_new.copy()
            break

        # Mixing for stability
        D = (1 - mixing) * D + mixing * D_new

        E_prev = E_tot

    else:
        print("WARNING: GHF did not converge within max_iter!")

    eigvals, eigvecs = diagonalize_fock(F)

    return F, D, eigvals # , scf_history


def get_impurity_indices(n_imp, n_bath):
    """Return the list of impurity orbital indices."""
    # Impurity orbitals: first n_imp orbitals, for spin up and down
    up_indices = list(range(0,2*n_imp,2))
    dn_indices = list(range(1, 2*n_imp+1,2))
    return up_indices, dn_indices

def build_mean_field(n_bath, n_imp, h_mat, D, U, J, Up):
    """
    Build the normal and anomalous mean-field potentials (Fock matrices).

    Parameters
    ----------
    D : ndarray
        Normal density matrix.
    P : ndarray
        Anomalous density matrix.

    Returns
    -------
    F : ndarray
        Normal Fock matrix.
    Delta : ndarray
        Anomalous Fock matrix.
    """
    norb = n_bath + n_imp

    F = np.zeros((2*norb, 2*norb), dtype=np.float64)

    # Get impurity indices
    up_imp, dn_imp = get_impurity_indices(n_bath=n_bath, n_imp=n_imp)

    # Hartree and Fock contributions from Slater-Kanamori
    # Only impurity orbitals have interactions

    F[up_imp, up_imp] += U * D[dn_imp, dn_imp].real
    F[dn_imp, dn_imp] += U * D[up_imp, up_imp].real

    F[up_imp, up_imp] += J * D[up_imp, up_imp].sum().real + (Up - J) * D[up_imp, up_imp].sum().real
    F[dn_imp, dn_imp] += J * D[dn_imp, dn_imp].sum().real + (Up - J) * D[dn_imp, dn_imp].sum().real

    F[up_imp, up_imp] -= J * D[up_imp, up_imp].real + (Up - J) * D[up_imp, up_imp].real
    F[dn_imp, dn_imp] -= J * D[dn_imp, dn_imp].real + (Up - J) * D[dn_imp, dn_imp].real

    # Add the one-body Hamiltonian
    F = F + h_mat

    return F

def diagonalize_fock(F):
    """
    Diagonalize the generalized Fock matrix.

    Parameters
    ----------
    F_GHF : ndarray
        Generalized Fock matrix.

    Returns
    -------
    eigvals : ndarray
        Eigenvalues sorted in ascending order.
    eigvecs : ndarray
        Corresponding eigenvectors.
    """
    eigvals, eigvecs = np.linalg.eigh(F)
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs

def compute_density_matrices(F, nocc, norb, kBT=1e-5, efermi0=0., ntol=1e-4):
    """
    Compute the normal and anomalous density matrices from occupied states.

    Parameters
    ----------
    F_GHF : ndarray
        Generalized Fock matrix.
    nocc : int
        Number of electrons to occupy.

    Returns
    -------
    D_new : ndarray
        Updated normal density matrix.
    P_new : ndarray
        Updated anomalous density matrix.
    """
    # Occupy the lowest nocc states
    # In Nambu space, each quasiparticle state can hold 2 electrons
    # However, to keep it simple, assume nocc <= dim
    # and occupy the lowest nocc states

    # Occupy states with negative eigenvalues (assuming symmetric spectrum)
    # This is more accurate for superconducting systems
    efermi = find_E_fermi(F[None,...], nocc=nocc, kBT=kBT, E_fermi0=efermi0, ntol=ntol)
    eigvals, eigvecs = diagonalize_fock(F=F)

    factor = fermi_dirac(eigvals, efermi, kBT)
    mask = factor > 1e-10
    factor = factor[mask]
    eigvecs = eigvecs[:,mask]

    D_new = (eigvecs[:norb*2] * factor[None,...]) @ eigvecs[:norb*2].conj().T
    
    # Ensure Hermiticity
    D_new = (D_new + D_new.T.conj()) / 2

    # assert np.abs(D_new.imag).max() < 1e-10, np.abs(D_new.imag).max()

    return D_new, efermi

def compute_energy(h_mat, F, D):
    """
    Compute the total energy.

    Parameters
    ----------
    F : ndarray
        Normal Fock matrix.
    Delta : ndarray
        Anomalous Fock matrix.
    D : ndarray
        Normal density matrix.
    P : ndarray
        Anomalous density matrix.

    Returns
    -------
    E_tot : float
        Total Hartree-Fock energy.
    """
    # Standard HF energy: E = Tr[D h] + 0.5 * Tr[D F]
    E_one_body = np.trace(D @ h_mat).real
    E_two_body = 0.5 * np.trace(D @ F).real
    E_tot = E_one_body + E_two_body
    return E_tot


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


