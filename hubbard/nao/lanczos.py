"""
    YinZhangHao Zhou <yinzhanghao.zhou@mail.mcgill.com>
    This code is modified from https://github.com/GiggleLiu/nrg_mapping/tree/master, the author is JinGuo Liu.
    The block tridiagonalization in stablized by discarding the direction of new transformation vector that have infinitesmal eigenvalues.
    In principle, this should work well on large block size.
"""

from numpy import sqrt, newaxis, dot, conj, array, zeros, ndim, append
import numpy as np
import scipy.sparse as sps
from scipy.linalg import qr, sqrtm, inv
import warnings

def icgs(u,Q,M=None,return_norm=False,maxiter=30):
    '''
    Iterative Classical M-orthogonal Gram-Schmidt orthogonalization.

    Parameters:
        :u: vector, the column vector to be orthogonalized.
        :Q: matrix, the search space.
        :M: matrix/None, the matrix, if provided, perform M-orthogonal.
        :return_norm: bool, return the norm of u.
        :maxiter: int, the maximum number of iteractions.

    Return:
        vector, orthogonalized vector u.
    '''
    assert(ndim(u)==2)
    uH,QH=u.T.conj(),Q.T.conj()
    alpha=0.5
    it=1
    Mu=M.dot(u) if M is not None else u
    r_pre=np.linalg.norm(uH.dot(Mu))
    for it in range(maxiter):
        u=u-Q.dot(QH.dot(Mu))
        Mu=M.dot(u) if M is not None else u
        r1=np.linalg.norm(uH.dot(Mu))
        if r1>alpha*r_pre:
            break
        r_pre=r1
    if r1<=alpha*r_pre:
        warnings.warn('loss of orthogonality @icgs.')
    return (u,r1) if return_norm else u

def tridiagonalize_sqrtm(A,q,m=None):
    """
    Use block lanczos algorithm to generate the tridiagonal part of matrix.
    This is the symmetric version of block-tridiagonalization in contrast to `qr` version.
    However, only matrices with blocksize p = 2 are currently supported.

    Parameters:
        :A: A sparse Hermitian matrix.
        :q: The starting columnwise orthogonal vector q with shape (n*p,p) with p the block size and n the number of blocks.
        :m: the steps to run.

    Return:
        **Note:** The orthogonality of initial vector q will be re-inforced to guarant the convergent result,
        meanwhile, the orthogonality of starting vector is also checked.
    """
    n=q.shape[1]
    if sps.issparse(A): A=A.toarray()
    if m is None:
        m = int(A.shape[0] // n)

    #check for othogonality of `q vector`.
    if not np.allclose(q.T.conj().dot(q),np.identity(q.shape[1])):
        raise Exception('Error','Othogoanlity check for start vector q failed.')
    #reinforce the orthogonality.
    Q=qr(q,mode='economic')[0]

    #initialize states
    alpha=[]
    beta=[]
    n_dynamic = [n]
    
    #run steps
    for i in range(m):
        if i == 0:
            qi_1 = Q
        else:
            qi_1 = Q[:, sum(n_dynamic[:i-1]):sum(n_dynamic[:i+1])]
        # qi_1=Q[:,(i-1)*n:(i+1)*n]
        qi=Q[:,i*n:i*n+n]

        z=A.dot(qi)
        ai=dot(z.T.conj(),qi)
        tmp=dot(qi_1.T.conj(),z)
        tmp=dot(qi_1,tmp)
        z=z-tmp #  A @ qi - qi_1 @ qi_1.T @ A @ qi

        # check z's zero svd value
        U, S, Vh = np.linalg.svd(z, False)
        mask = S > 1e-8
        if not np.all(mask):
            # U_, S_, Vh_ = np.linalg.svd(np.random.randn(U.shape[0],np.sum), False)
            new_col = icgs(np.random.randn(A.shape[0],np.sum(~mask)), np.concat([Q,U[:,mask]], axis=1))
            U[:,~mask] = new_col
            z = U

        n_dynamic.append(z.shape[1])
        

        bi=sqrtm(dot(z.T.conj(),z))
        alpha.append(ai)
        beta.append(bi)

        if i==m-1: break
        z=dot(z,inv(bi))
        Q_i=icgs(z,Q)
        Q=append(Q,Q_i,axis=-1)

        if np.sum(abs(bi))<1e-20:
            print('Warning! bad krylov space!')
    
    return Q