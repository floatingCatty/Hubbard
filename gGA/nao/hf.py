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
    dn_indices = list(range(n_bath+n_imp, n_bath+n_imp + n_imp))
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
    efermi = find_E_fermi(F_GHF[None,...], nocc=nocc, kBT=kBT, E_fermi0=efermi0, ntol=ntol)
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


def hartree_fock_slater_kanamori(
    h_mat,
    n_imp,
    n_bath,
    U,
    Up,
    J,
    Jp,
    num_electrons,
    max_iter=50,
    tol=1e-6
):
    """
    Perform a Hartree-Fock calculation for an impurity+bath system 
    with Slater-Kanamori interactions only on the impurity orbitals.
    
    Parameters
    ----------
    h_mat : (2*(n_imp + n_bath), 2*(n_imp + n_bath)) ndarray
        One-body Hamiltonian (spin included).
    n_imp : int
        Number of impurity orbitals (spin-orbitals are 2*n_imp total).
    n_bath : int
        Number of bath orbitals (spin-orbitals are 2*n_bath total).
    U, Up, J, Jp : float
        Slater-Kanamori parameters:
          - U  : intra-orbital
          - Up : inter-orbital
          - J  : Hund's coupling
          - Jp : pair hopping (not fully included in the simplest HF density-density form,
                 but can be used or extended in advanced decouplings)
    num_electrons : int
        Total number of electrons to be placed in the system.
    max_iter : int
        Maximum number of SCF iterations.
    tol : float
        Convergence tolerance on density matrix difference.
    
    Returns
    -------
    F_total : ndarray
        Final Fock matrix (mean-field Hamiltonian).
    D : ndarray
        Final density matrix.
    eigvals : ndarray
        Final eigenvalues of F_total.
    scf_history : list
        List of tuples (iteration, energy, rms_density_diff).
    """
    # total number of (spin-)orbitals
    n_orb = n_imp + n_bath
    dim = 2 * n_orb  # spin up + spin down
    
    # --- Helper indexing functions ---
    # We define an orbital index as orb_index(spin, orbital) 
    # if we want to be explicit. Or we can just use:
    #   up   = 0..(n_orb-1),
    #   down = n_orb..(2*n_orb-1).
    
    def get_occupations_from_density(D):
        """
        Return occupation numbers n_{m, spin} for m in 0..(n_imp-1) (impurity part)
        and also for the entire system if needed. 
        We only need impurity block to build the Slater-Kanamori potential.
        
        D is the full density matrix of shape (dim, dim).
        """
        # Occupation for orbital i is simply D[i,i] (since we assume real/diagonal in that basis).
        occ_up_imp   = [D[i, i].real for i in range(n_imp)]  # spin up, impurity orbitals
        occ_down_imp = [D[n_orb + i, n_orb + i].real for i in range(n_imp)]  # spin down, impurity orbitals
        
        return np.array(occ_up_imp), np.array(occ_down_imp)
    
    def build_slater_kanamori_mf_potential(D):
        """
        Build the mean-field potential matrix \Sigma_MF from Slater-Kanamori 
        interactions on the impurity orbitals, using a density-density decoupling.
        
        We'll return a (dim, dim) matrix, but only the first 2*n_imp diagonal 
        elements (the impurity block) are nonzero (the bath block has zero interaction).
        """
        Sigma = np.zeros((dim, dim), dtype=np.float64)
        
        # get impurity occupations
        occ_up_imp, occ_down_imp = get_occupations_from_density(D)
        
        # For each impurity orbital m, spin sigma, we compute:
        # Sigma(m, up)   = U * n_{m,down} 
        #               + sum_{m' != m} [ (Up - J) * n_{m',up} + Up * n_{m',down} ]
        #
        # Sigma(m, down) = U * n_{m,up}
        #               + sum_{m' != m} [ (Up - J) * n_{m',down} + Up * n_{m',up} ]
        #
        # We ignore pair-hopping or spin-flip contributions in the simplest density-density approach, 
        # but you can incorporate them if needed.
        
        for m in range(n_imp):
            # up-channel
            diag_up = ( U * occ_down_imp[m] ) 
            sum_others_up = 0.0
            for mprime in range(n_imp):
                if mprime != m:
                    sum_others_up += (Up - J)*occ_up_imp[mprime] + Up*occ_down_imp[mprime]
            diag_up += sum_others_up
            
            # down-channel
            diag_down = ( U * occ_up_imp[m] )
            sum_others_down = 0.0
            for mprime in range(n_imp):
                if mprime != m:
                    sum_others_down += (Up - J)*occ_down_imp[mprime] + Up*occ_up_imp[mprime]
            diag_down += sum_others_down
            
            # place on the diagonal of Sigma
            Sigma[m, m] = diag_up
            Sigma[n_orb + m, n_orb + m] = diag_down
            
        return Sigma
    
    def diagonalize_and_fill(F):
        """
        Diagonalize the Fock matrix F, fill the lowest num_electrons states,
        and build the density matrix D. Returns (eigenvalues, D).
        """
        # Diagonalize
        eigvals, eigvecs = la.eigh(F)
        
        # Occupy the lowest num_electrons eigenstates
        # (assuming T=0, spin-orbitals are each singly occupied in ascending eigenvalue order)
        idx_occupied = np.argsort(eigvals)[:num_electrons]
        
        # Build density matrix from occupied orbitals
        # D = sum_{k in occupied} |phi_k><phi_k|
        D_new = np.dot(eigvecs[:, idx_occupied], eigvecs[:, idx_occupied].T.conj())
        
        return eigvals, D_new
    
    def hf_energy(F, D, h):
        """
        Compute the HF total energy:
          E = 0.5 * sum_{ij} D_{ji} (h_{ij} + F_{ij})
        The factor of 0.5 is used because F = h + V_HF and we don't want to double-count.
        """
        return 0.5 * np.sum(D.T * (h + F)).real
    
    # -----------------
    # Start SCF
    # -----------------
    scf_history = []
    
    # Initial guess for density: fill uniformly or set to zero for bath, etc.
    # For simplicity, fill each orbital equally with n = num_electrons / dim
    # This is just one possible simple guess.
    initial_occup = float(num_electrons) / float(dim)
    D = np.eye(dim) * initial_occup
    
    # SCF iteration
    for iteration in range(max_iter):
        # Build the HF potential from Slater–Kanamori
        Sigma_int = build_slater_kanamori_mf_potential(D)
        
        # Build the Fock matrix
        F_total = h_mat + Sigma_int
        
        # Diagonalize and get new density
        eigvals, D_new = diagonalize_and_fill(F_total)
        
        # Compute SCF metrics
        d_diff = np.linalg.norm(D_new - D)
        E_tot = hf_energy(F_total, D_new, h_mat)
        
        scf_history.append((iteration, E_tot, d_diff))
        print(f"Iteration {iteration:3d}: E = {E_tot: .6f}, dD = {d_diff: .6e}")
        
        # Check convergence
        if d_diff < tol:
            D = D_new
            break
        
        D = D_new
    
    else:
        print("WARNING: SCF did not converge within max_iter!")
    
    return F_total, D, eigvals, scf_history