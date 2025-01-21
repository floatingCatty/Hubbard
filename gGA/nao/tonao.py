import numpy as np
import scipy.linalg as la
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

        assert np.abs(D[:,1,:,0]).max() < 1e-12
    else:
        sfactor = 2

    if sfactor == 2:
        h_mat_trans, D_trans, trans_mat_nao = trans_nao(h_mat, D, n_imp=n_imp, n_bath=n_bath, sfactor=sfactor)
        h_mat_trans, D_trans, trans_mat_bonding = trans_bonding(h_mat_trans, D_trans, n_imp, n_bath, sfactor=sfactor)
        h_mat_trans, D_trans, trans_mat_chain = construct_chain(h_mat_trans, D_trans, trans_mat_bonding, n_imp, sfactor=sfactor)

        trans_mat = trans_mat_chain @ trans_mat_bonding @ trans_mat_nao

        print("error for H: ", np.abs(trans_mat.conj().T @ h_mat_trans @ trans_mat - h_mat).max())
        print("error for D: ", np.abs(trans_mat.conj().T @ D_trans @ trans_mat - D).max())
        print("error for imp: ", np.abs(h_mat_trans[:n_imp*2,:n_imp*2]-h_mat[:n_imp*2,:n_imp*2]).max())
    else:
        h_mat_trans_up, D_trans_up, trans_mat_nao_up = trans_nao(h_mat_up, D_up, n_imp=n_imp, n_bath=n_bath, sfactor=sfactor)
        h_mat_trans_up, D_trans_up, trans_mat_bonding_up = trans_bonding(h_mat_trans_up, D_trans_up, n_imp, n_bath, sfactor=sfactor)
        h_mat_trans_up, D_trans_up, trans_mat_chain_up = construct_chain(h_mat_trans_up, D_trans_up, trans_mat_bonding_up, n_imp, sfactor=sfactor)

        trans_mat_up = trans_mat_chain_up @ trans_mat_bonding_up @ trans_mat_nao_up

        print("error for H up: ", np.abs(trans_mat_up.conj().T @ h_mat_trans_up @ trans_mat_up - h_mat_up).max())
        print("error for D up: ", np.abs(trans_mat_up.conj().T @ D_trans_up @ trans_mat_up - D_up).max())
        print("error for imp up: ", np.abs(h_mat_trans_up[:n_imp,:n_imp]-h_mat_up[:n_imp,:n_imp]).max())

        h_mat_trans_dn, D_trans_dn, trans_mat_nao_dn = trans_nao(h_mat_dn, D_dn, n_imp=n_imp, n_bath=n_bath, sfactor=sfactor)
        h_mat_trans_dn, D_trans_dn, trans_mat_bonding_dn = trans_bonding(h_mat_trans_dn, D_trans_dn, n_imp, n_bath, sfactor=sfactor)
        h_mat_trans_dn, D_trans_dn, trans_mat_chain_dn = construct_chain(h_mat_trans_dn, D_trans_dn, trans_mat_bonding_dn, n_imp, sfactor=sfactor)

        trans_mat_dn = trans_mat_chain_dn @ trans_mat_bonding_dn @ trans_mat_nao_dn

        print("error for H dn: ", np.abs(trans_mat_dn.conj().T @ h_mat_trans_dn @ trans_mat_dn - h_mat_dn).max())
        print("error for D dn: ", np.abs(trans_mat_dn.conj().T @ D_trans_dn @ trans_mat_dn - D_dn).max())
        print("error for imp dn: ", np.abs(h_mat_trans_dn[:n_imp,:n_imp]-h_mat_dn[:n_imp,:n_imp]).max())

        trans_mat = la.block_diag(trans_mat_up, trans_mat_dn).reshape(2,norb,2,norb).transpose(1,0,3,2).reshape(2*norb,2*norb)
        h_mat_trans = la.block_diag(h_mat_trans_up, h_mat_trans_dn).reshape(2,norb,2,norb).transpose(1,0,3,2).reshape(2*norb,2*norb)
        D_trans = la.block_diag(D_trans_up, D_trans_dn).reshape(2,norb,2,norb).transpose(1,0,3,2).reshape(2*norb,2*norb)
        
        print("error for H: ", np.abs(trans_mat.conj().T @ h_mat_trans @ trans_mat - h_mat).max())
        print("error for D: ", np.abs(trans_mat.conj().T @ D_trans @ trans_mat - D).max())
        print("error for imp: ", np.abs(h_mat_trans[:n_imp*2,:n_imp*2]-h_mat[:n_imp*2,:n_imp*2]).max())
        
    return h_mat_trans, D_trans, trans_mat

def trans_nao(h_mat, D, n_imp, n_bath, sfactor=2):
    norb = n_imp + n_bath
    dim = norb * sfactor

    assert h_mat.shape[0] == h_mat.shape[1] == dim
    assert D.shape[0] == D.shape[1] == dim

    bath = D[-n_bath*sfactor:,-n_bath*sfactor:]

    eigvals, eigvecs = la.eigh(bath)
    int_bath_mask = (eigvals > 1e-12) * (eigvals < (1-1e-12))
    eigvals[int_bath_mask] = -1.
    sort_index = np.argsort(eigvals)
    eigvals = eigvals[sort_index]
    eigvecs = eigvecs[:,sort_index]

    trans_mat = eigvecs.conj().T
    trans_mat = la.block_diag(np.eye(n_imp*sfactor), trans_mat)

    D_trans = trans_mat @ D @ trans_mat.conj().T
    h_mat_trans = trans_mat @ h_mat @ trans_mat.conj().T

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
    trans_mat = h_mat_trans[:2*sfactor*n_imp, :2*sfactor*n_imp] # is it correct?

    
    eigvals, eigvecs = np.linalg.eigh(trans_mat)
    sort_index = np.argsort(eigvals)
    trans_mat = eigvecs[:,sort_index].conj().T

    trans_mat = la.block_diag(trans_mat, np.eye((n_bath-n_imp)*sfactor))

    D_trans = trans_mat @ D_trans @ trans_mat.conj().T
    h_mat_trans = trans_mat @ h_mat_trans @ trans_mat.conj().T
    
    return h_mat_trans, D_trans, trans_mat


def construct_chain(h_mat_trans, D_trans, bond_trans_mat, n_imp, sfactor=2):
    # it need a block lanzco algorithm
    """
        
    """
    D_diag = np.diag(D_trans)
    # first, check the number of bonding and anti-bonding states
    n_anti = np.sum(D_diag[:2*sfactor*n_imp] < 1e-14)
    n_bond = np.sum(~(D_diag[:2*sfactor*n_imp] < 1e-14))

    assert n_anti > 0
    assert n_bond > 0

    empty_index = np.arange(D_diag.shape[0])[D_diag<1e-14]
    full_index = np.arange(D_diag.shape[0])[~(D_diag<1e-14)]

    # we need to check whether empty states's number can divide n_anti
    # and full states's number can divide n_bond

    assert len(full_index) % n_bond == 0
    assert len(empty_index) % n_anti == 0

    h_empty_block = h_mat_trans[np.ix_(empty_index, empty_index)]
    h_full_block = h_mat_trans[np.ix_(full_index, full_index)]
    
    p = np.zeros((len(empty_index), n_anti))
    p[:n_anti] = np.eye(n_anti)
    _, _, Q_empty = tridiagonalize_sqrtm(h_empty_block, p, None, True)

    p = np.zeros((len(full_index), n_bond))
    p[:n_bond] = np.eye(n_bond)
    _, _, Q_full = tridiagonalize_sqrtm(h_full_block, p, None, True)

    Q = np.eye(D_trans.shape[0])
    Q[np.ix_(empty_index, empty_index)] = Q_empty
    Q[np.ix_(full_index, full_index)] = Q_full
    Q = Q.T.conj()
    Q = bond_trans_mat.conj().T @ Q

    return Q @ h_mat_trans @ Q.conj().T, Q @ D_trans @ Q.T.conj(), Q