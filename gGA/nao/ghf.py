import numpy as np
import scipy.linalg as la
from gGA.gutz.kinetics import fermi_dirac, find_E_fermi

def generalized_hartree_fock(
    h_mat,
    n_imp,
    n_bath,
    nocc,
    U,
    Uprime,
    J,
    Jprime,
    max_iter=50,
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
    Uprime : float
        Inter-orbital Coulomb interaction.
    J : float
        Hund's rule coupling (exchange).
    Jprime : float
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
    P = np.zeros((dim, dim), dtype=np.float64)  # No initial pairing

    # scf_history = []
    E_prev = 0.0
    efermi = 0.

    for iteration in range(max_iter):
        # Build mean-field potentials
        F, Delta = build_mean_field(n_imp=n_imp, n_bath=n_bath, h_mat=h_mat, D=D, P=P, U=U, J=J, Uprime=Uprime, Jprime=Jprime)

        # Construct generalized Fock matrix
        F_GHF = build_generalized_fock(F, Delta)

        # Compute new density matrices
        D_new, P_new, efermi = compute_density_matrices(F_GHF=F_GHF, nocc=nocc, norb=n_orb, kBT=kBT, efermi0=efermi, ntol=ntol)

        # Compute energy
        E_tot = compute_energy(h_mat, F, Delta, D_new, P_new)

        # Compute density difference
        dD = np.linalg.norm(D_new - D)
        dP = np.linalg.norm(P_new - P)
        d = np.sqrt(dD**2 + dP**2)

        # scf_history.append((iteration, E_tot, d))

        if verbose:
            print(f"Iter {iteration:3d}: E = {E_tot:.6f}, ΔD,P = {d:.6e}")

        # Check convergence
        if d < tol:
            D = D_new.copy()
            P = P_new.copy()
            break

        # Mixing for stability
        D = (1 - mixing) * D + mixing * D_new
        P = (1 - mixing) * P + mixing * P_new

        E_prev = E_tot

    else:
        if verbose:
            print("WARNING: GHF did not converge within max_iter!")

    eigvals, eigvecs = diagonalize_fock(F_GHF)

    return F, Delta, D, P, eigvals # , scf_history

def build_generalized_fock(F, Delta):
    """
    Construct the generalized Fock matrix in Nambu space.

    Parameters
    ----------
    F : ndarray
        Normal Fock matrix.
    Delta : ndarray
        Anomalous Fock matrix.

    Returns
    -------
    F_GHF : ndarray
        Generalized Fock matrix of shape (2*dim, 2*dim).
    """
    upper = np.hstack((F, Delta))
    lower = np.hstack((Delta.conj(), -F.conj()))
    F_GHF = np.vstack((upper, lower))
    return F_GHF

def get_impurity_indices(n_imp, n_bath):
    """Return the list of impurity orbital indices."""
    # Impurity orbitals: first n_imp orbitals, for spin up and down
    up_indices = list(range(n_imp))
    dn_indices = list(range(n_imp, 2*n_imp))
    return up_indices, dn_indices

def build_mean_field(n_bath, n_imp, h_mat, D, P, U, J, Uprime, Jprime):
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
    Delta = np.zeros((2*norb, 2*norb), dtype=np.float64)

    # Get impurity indices
    up_imp, dn_imp = get_impurity_indices(n_bath=n_bath, n_imp=n_imp)

    # Hartree and Fock contributions from Slater-Kanamori
    # Only impurity orbitals have interactions

    F[up_imp, up_imp] += U * D[dn_imp, dn_imp]
    F[dn_imp, dn_imp] += U * D[up_imp, up_imp]

    F[up_imp, up_imp] += J * D[up_imp, up_imp].sum() + (Uprime - J) * D[up_imp, up_imp].sum()
    F[dn_imp, dn_imp] += J * D[dn_imp, dn_imp].sum() + (Uprime - J) * D[dn_imp, dn_imp].sum()

    F[up_imp, up_imp] -= J * D[up_imp, up_imp] + (Uprime - J) * D[up_imp, up_imp]
    F[dn_imp, dn_imp] -= J * D[dn_imp, dn_imp] + (Uprime - J) * D[dn_imp, dn_imp]

    Delta[up_imp, dn_imp] += Jprime * P[up_imp, dn_imp].sum()
    Delta[dn_imp, up_imp] += Jprime * P[dn_imp, up_imp].sum()

    Delta[up_imp, dn_imp] -= Jprime * P[up_imp, dn_imp]
    Delta[dn_imp, up_imp] -= Jprime * P[dn_imp, up_imp]

    # Add the one-body Hamiltonian
    F += h_mat

    return F, Delta

def diagonalize_fock(F_GHF):
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
    eigvals, eigvecs = la.eigh(F_GHF)
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs

def compute_density_matrices(F_GHF, nocc, norb, kBT=1e-5, efermi0=0., ntol=1e-4):
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
    efermi = find_E_fermi(F_GHF[None,...], nocc=2*nocc, kBT=kBT, E_fermi0=efermi0, ntol=ntol)
    eigvals, eigvecs = diagonalize_fock(F_GHF=F_GHF)

    factor = fermi_dirac(eigvals, efermi, kBT)
    mask = factor > 1e-10
    factor = factor[mask]
    eigvecs = eigvecs[:,mask]

    D_new = (eigvecs[:norb*2] * factor[None,...]) @ eigvecs[:norb*2].conj().T
    P_new = (eigvecs[:norb*2] * factor[None,...]) @ eigvecs[norb*2:].conj().T
    
    # Ensure Hermiticity
    D_new = (D_new + D_new.T.conj()) / 2
    P_new = (P_new + P_new.T.conj()) / 2

    print(len(factor), np.trace(D_new), eigvals, np.trace((eigvecs[-norb*2:] * factor[None,...]) @ eigvecs[-norb*2:].conj().T))

    return D_new.real, P_new.real, efermi

def compute_energy(h_mat, F, Delta, D, P):
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
    # Contribution from pairing (if any)
    E_pair = 0.5 * np.trace(P @ Delta).real
    E_tot = E_one_body + E_two_body + E_pair
    return E_tot


