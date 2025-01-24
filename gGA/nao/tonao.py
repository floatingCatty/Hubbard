import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from gGA.nao.lanczos import tridiagonalize_sqrtm

# TODO: add nspin dependency, it should be easy, since nspin=1,2 only requires to split the up and down independently, and 
# nspin = 4 is naturally supported by current code.

def nao_two_chain(h_mat, D, n_imp, n_bath, nspin=1):
    norb = n_imp + n_bath
    if nspin <= 2:
        sfactor = 1
        h_mat = h_mat.reshape(norb, 2, norb, 2)
        h_mat_up = h_mat[:,0,:,0]
        h_mat_dn = h_mat[:,1,:,1]

        assert np.abs(h_mat[:,1,:,0]).max() < 1e-12

        D = D.reshape(norb, 2, norb, 2)
        D_up = D[:,0,:,0]
        D_dn = D[:,1,:,1]

        print("occ for up spin: {:.1f}, occ for down spin: {:.1f}".format(np.trace(D_up), np.trace(D_dn)))

        assert np.abs(D[:,1,:,0]).max() < 1e-12

        h_mat = h_mat.reshape(norb*2, norb*2)
        D = D.reshape(norb*2, norb*2)
    else:
        sfactor = 2

    if sfactor == 2:
        # import matplotlib.pyplot as plt

        h_mat_trans, D_trans, trans_mat_nao = trans_nao(h_mat, D, n_imp=n_imp, n_bath=n_bath, sfactor=sfactor)
        recovered_D = trans_mat_nao.conj().T @ D_trans @ trans_mat_nao
        recovered_H = trans_mat_nao.conj().T @ h_mat_trans @ trans_mat_nao
        print("### Transformation NAO:")
        print("error for H: ", np.abs(recovered_H - h_mat).max())
        print("error for D: ", np.abs(recovered_D - D).max())
        NAO_D = D_trans.copy()
        NAO_H = h_mat_trans.copy()

        h_mat_trans, D_trans, trans_mat_bonding = trans_bonding(h_mat_trans, D_trans, n_imp, n_bath, sfactor=sfactor)

        print("### Transformation Bonding:")
        recovered_D = trans_mat_bonding.conj().T @ D_trans @ trans_mat_bonding 
        recovered_H = trans_mat_bonding.conj().T @ h_mat_trans @ trans_mat_bonding
        print("error for H: ", np.abs(recovered_H - NAO_H).max())
        print("error for D: ", np.abs(recovered_D - NAO_D).max())

        BD_D = D_trans.copy()
        BD_H = h_mat_trans.copy()

        h_mat_trans, D_trans, trans_mat_chain = construct_chain(h_mat_trans, D_trans, trans_mat_bonding, n_imp, sfactor=sfactor)
        trans_mat = trans_mat_chain @ trans_mat_bonding @ trans_mat_nao

        recovered_D = trans_mat_chain.conj().T @ D_trans @ trans_mat_chain
        recovered_H = trans_mat_chain.conj().T @ h_mat_trans @ trans_mat_chain

        print("### Transformation Chain:")
        print("error for H: ", np.abs(recovered_H - BD_H).max())
        print("error for D: ", np.abs(recovered_D - BD_D).max())
        print("error for imp: ", np.abs(h_mat_trans[:n_imp*2,:n_imp*2]-h_mat[:n_imp*2,:n_imp*2]).max())
    else:
        h_mat_trans_up, D_trans_up, trans_mat_nao_up = trans_nao(h_mat_up, D_up, n_imp=n_imp, n_bath=n_bath, sfactor=sfactor)
        recovered_D = trans_mat_nao_up.conj().T @ D_trans_up @ trans_mat_nao_up
        recovered_H = trans_mat_nao_up.conj().T @ h_mat_trans_up @ trans_mat_nao_up
        print("### Transformation NAO up:")
        print("error for H: ", np.abs(recovered_H - h_mat_up).max())
        print("error for D: ", np.abs(recovered_D - D_up).max())
        NAO_D = D_trans_up.copy()
        NAO_H = h_mat_trans_up.copy()

        h_mat_trans_up, D_trans_up, trans_mat_bonding_up = trans_bonding(h_mat_trans_up, D_trans_up, n_imp, n_bath, sfactor=sfactor)

        print("### Transformation Bonding up:")
        recovered_D = trans_mat_bonding_up.conj().T @ D_trans_up @ trans_mat_bonding_up
        recovered_H = trans_mat_bonding_up.conj().T @ h_mat_trans_up @ trans_mat_bonding_up
        print("error for H: ", np.abs(recovered_H - NAO_H).max())
        print("error for D: ", np.abs(recovered_D - NAO_D).max())

        BD_D = D_trans_up.copy()
        BD_H = h_mat_trans_up.copy()

        h_mat_trans_up, D_trans_up, trans_mat_chain_up = construct_chain(h_mat_trans_up, D_trans_up, trans_mat_bonding_up, n_imp, sfactor=sfactor)
        trans_mat_up = trans_mat_chain_up @ trans_mat_bonding_up @ trans_mat_nao_up

        recovered_D = trans_mat_chain_up.conj().T @ D_trans_up @ trans_mat_chain_up
        recovered_H = trans_mat_chain_up.conj().T @ h_mat_trans_up @ trans_mat_chain_up

        print("### Transformation Chain up:")
        print("error for H: ", np.abs(recovered_H - BD_H).max())
        print("error for D: ", np.abs(recovered_D - BD_D).max())
        print("error for imp: ", np.abs(h_mat_trans_up[:n_imp,:n_imp]-h_mat_up[:n_imp,:n_imp]).max())

        h_mat_trans_dn, D_trans_dn, trans_mat_nao_dn = trans_nao(h_mat_dn, D_dn, n_imp=n_imp, n_bath=n_bath, sfactor=sfactor)
        h_mat_trans_dn, D_trans_dn, trans_mat_bonding_dn = trans_bonding(h_mat_trans_dn, D_trans_dn, n_imp, n_bath, sfactor=sfactor)
        h_mat_trans_dn, D_trans_dn, trans_mat_chain_dn = construct_chain(h_mat_trans_dn, D_trans_dn, trans_mat_bonding_dn, n_imp, sfactor=sfactor)

        trans_mat_dn = trans_mat_chain_dn @ trans_mat_bonding_dn @ trans_mat_nao_dn

        print("error for H dn: ", np.abs(trans_mat_dn.conj().T @ h_mat_trans_dn @ trans_mat_dn - h_mat_dn).max())
        print("error for D dn: ", np.abs(trans_mat_dn.conj().T @ D_trans_dn @ trans_mat_dn - D_dn).max())
        print("error for imp dn: ", np.abs(h_mat_trans_dn[:n_imp,:n_imp]-h_mat_dn[:n_imp,:n_imp]).max())

        trans_mat = sla.block_diag(trans_mat_up, trans_mat_dn).reshape(2,norb,2,norb).transpose(1,0,3,2).reshape(2*norb,2*norb)
        h_mat_trans = sla.block_diag(h_mat_trans_up, h_mat_trans_dn).reshape(2,norb,2,norb).transpose(1,0,3,2).reshape(2*norb,2*norb)
        D_trans = sla.block_diag(D_trans_up, D_trans_dn).reshape(2,norb,2,norb).transpose(1,0,3,2).reshape(2*norb,2*norb)
        
        print("error for H: ", np.abs(trans_mat.conj().T @ h_mat_trans @ trans_mat - h_mat).max())
        print("error for D: ", np.abs(trans_mat.conj().T @ D_trans @ trans_mat - D).max())
        print("error for imp: ", np.abs(h_mat_trans[:n_imp*2,:n_imp*2]-h_mat[:n_imp*2,:n_imp*2]).max())

    return h_mat_trans, D_trans, trans_mat

def trans_nao(h_mat, D, n_imp, n_bath, sfactor=2):
    norb = n_imp + n_bath
    dim = norb * sfactor

    assert h_mat.shape[0] == h_mat.shape[1] == dim
    assert D.shape[0] == D.shape[1] == dim

    
    bath_H = h_mat[-n_bath*sfactor:,-n_bath*sfactor:]

    # first we diagonalize the bath orbital
    eigvals, eigvecs = la.eigh(bath_H)
    trans_mat_decouple = sla.block_diag(np.eye(n_imp*sfactor), eigvecs.conj().T)
    
    D_trans = trans_mat_decouple @ D @ trans_mat_decouple.conj().T
    h_mat_trans = trans_mat_decouple @ h_mat @ trans_mat_decouple.conj().T

    bath = D_trans[-n_bath*sfactor:,-n_bath*sfactor:]


    eigvals, eigvecs = la.eigh(bath)
    int_bath_mask = (eigvals > 1e-12) * (eigvals < (1-1e-12))
    eigvals[int_bath_mask] = -1.
    sort_index = np.argsort(eigvals)
    eigvals = eigvals[sort_index]
    eigvecs = eigvecs[:,sort_index]

    trans_mat = eigvecs.conj().T
    trans_mat = sla.block_diag(np.eye(n_imp*sfactor), trans_mat)

    D_trans = trans_mat @ D_trans @ trans_mat.conj().T
    h_mat_trans = trans_mat @ h_mat_trans @ trans_mat.conj().T

    trans_mat = trans_mat @ trans_mat_decouple

    return h_mat_trans, D_trans, trans_mat

def trans_bonding(h_mat_trans, D_trans, n_imp, n_bath, sfactor=2):
    # we assert the int bath is connected with imp

    imp_index = np.arange(n_imp*sfactor)

    trans_mat = np.eye((n_imp*sfactor)*2)
    mixing_bath_in_transmat_index = imp_index + n_imp
    # trans_mat[imp_index, imp_index] = D_trans[imp_index,imp_index]
    # trans_mat[mixing_bath_in_transmat_index, mixing_bath_in_transmat_index] = D_trans[mixing_bath_in_transmat_index, mixing_bath_in_transmat_index]
    # trans_mat[imp_index, mixing_bath_in_transmat_index] = D_trans[imp_index, mixing_bath_in_transmat_index]
    # trans_mat[mixing_bath_in_transmat_index, imp_index] = D_trans[mixing_bath_in_transmat_index, imp_index]
    trans_mat = D_trans[:2*sfactor*n_imp, :2*sfactor*n_imp] # is it correct?
    # trans_mat = h_mat_trans[:2*sfactor*n_imp, :2*sfactor*n_imp]

    
    eigvals, eigvecs = np.linalg.eigh(trans_mat)
    sort_index = np.argsort(-eigvals)
    trans_mat = eigvecs[:,sort_index].conj().T
    # trans_mat = eigvecs.conj().T

    trans_mat = sla.block_diag(trans_mat, np.eye((n_bath-n_imp)*sfactor))

    D_trans = trans_mat @ D_trans @ trans_mat.conj().T
    h_mat_trans = trans_mat @ h_mat_trans @ trans_mat.conj().T
    
    return h_mat_trans, D_trans, trans_mat


def construct_chain(h_mat_trans, D_trans, bond_trans_mat, n_imp, sfactor=2):
    # it need a block lanzco algorithm
    """
        
    """
    D_diag = np.diag(D_trans).real
    # first, check the number of bonding and anti-bonding states
    n_anti = np.sum(D_diag[:2*sfactor*n_imp] < 1e-7)
    n_bond = np.sum(~(D_diag[:2*sfactor*n_imp] < 1e-7))

    assert n_anti > 0, D_diag[:2*sfactor*n_imp]
    assert n_bond > 0, D_diag[:2*sfactor*n_imp]

    empty_index = np.arange(D_diag.shape[0])[D_diag<1e-7]
    full_index = np.arange(D_diag.shape[0])[~(D_diag<1e-7)]

    # we need to check whether empty states's number can divide n_anti
    # and full states's number can divide n_bond

    assert len(full_index) % n_bond == 0, "full_index: {}, n_bond: {}, occ: {}".format(len(full_index), n_bond, D_diag[:2*sfactor*n_imp])
    assert len(empty_index) % n_anti == 0, "empty_index: {}, n_bond: {}, occ: {}".format(len(empty_index), n_anti, D_diag[:2*sfactor*n_imp])

    h_empty_block = h_mat_trans[np.ix_(empty_index, empty_index)]
    h_full_block = h_mat_trans[np.ix_(full_index, full_index)]
    
    p = np.zeros((len(empty_index), n_anti))
    p[:n_anti] = np.eye(n_anti)
    Q_empty = tridiagonalize_sqrtm(h_empty_block, p, None)

    p = np.zeros((len(full_index), n_bond))
    p[:n_bond] = np.eye(n_bond)
    Q_full = tridiagonalize_sqrtm(h_full_block, p, None)

    Q = np.eye(D_trans.shape[0]).astype(Q_empty.dtype)
    Q[np.ix_(empty_index, empty_index)] = Q_empty
    Q[np.ix_(full_index, full_index)] = Q_full
    Q = Q.T.conj()
    Q = bond_trans_mat.conj().T @ Q

    return Q @ h_mat_trans @ Q.conj().T, Q @ D_trans @ Q.T.conj(), Q


if __name__ == "__main__":
    from gGA.nao.ghf import generalized_hartree_fock # not useful yet
    from gGA.nao.hf import hartree_fock
    import numpy as np
    import scipy.linalg as sla

    nimp = 1
    nbath = 15
    nspin = 4
    U = 5.
    J = 0.25
    Jp = J
    Up = U-2*J

    nocc = nimp + nbath
    norb = nocc

    if nspin == 1:
        H = np.random.randn(norb, norb)
        H += H.T
        # eigvals, eigvecs = sla.eigh(H[nimp:,nimp:])
        # H[nimp:] = eigvecs.conj().T @ H[nimp:]
        # H[:,nimp:] = H[:, nimp:] @ eigvecs
        H = np.kron(H, np.eye(2))
    elif nspin == 2:
        H_up = np.random.randn(norb, norb)
        H_up += H_up.T
        # eigvals, eigvecs = sla.eigh(H_up[nimp:,nimp:])
        # H_up[nimp:] = eigvecs.conj().T @ H_up[nimp:]
        # H_up[:,nimp:] = H_up[:, nimp:] @ eigvecs

        H_dn = np.random.randn(norb, norb)
        H_dn += H_dn.T
        # eigvals, eigvecs = sla.eigh(H_dn[nimp:,nimp:])
        # H_dn[nimp:] = eigvecs.conj().T @ H_dn[nimp:]
        # H_dn[:,nimp:] = H_dn[:, nimp:] @ eigvecs

        H = sla.block_diag(H_up, H_dn).reshape(2, norb, 2, norb).transpose(1,0,3,2).reshape(norb*2, norb*2)
    elif nspin == 4:
        H = np.random.randn(norb*2, norb*2)
        H += H.T

        # eigvals, eigvecs = sla.eigh(H[nimp*2:,nimp*2:])
        # H[nimp*2:] = eigvecs.conj().T @ H[nimp*2:]
        # H[:,nimp*2:] = H[:, nimp*2:] @ eigvecs

    else:
        raise ValueError



    # F, _, D, P, _ = generalized_hartree_fock(
    #     h_mat=H,
    #     n_imp=nimp,
    #     n_bath=nbath,
    #     nocc=nocc,
    #     U=U,
    #     J=J,
    #     Uprime=Up,
    #     Jprime=Jp,
    #     max_iter=100,
    #     ntol=1e-8,

    # )

    F,  D, _ = hartree_fock(
        h_mat=H,
        n_imp=nimp,
        n_bath=nbath,
        nocc=nocc,
        U=U,
        J=J,
        Uprime=Up,
        Jprime=Jp,
        max_iter=100,
        ntol=1e-8,
    )

    # check if GHF break spin setting
    D = D.reshape(norb,2,norb,2)

    if nspin<4:
        assert np.abs(D[:,0,:,1]).max() < 1e-9

    if nspin == 1:
        err = np.abs(D[:,0,:,0]-D[:,1,:,1]).max()
        assert err < 1e-9, "spin symmetry breaking error {}".format(err)
    D = D.reshape(norb*2, norb*2)

    h_mat_trans, D_trans, trans_mat = nao_two_chain(
                                    h_mat=F,
                                    D=D,
                                    n_imp=nimp,
                                    n_bath=nbath,
                                    nspin=nspin,
                                )