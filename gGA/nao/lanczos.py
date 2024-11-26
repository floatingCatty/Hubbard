from numpy import sqrt, newaxis, dot, conj, array, zeros, ndim, append
import numpy as np
import scipy.sparse as sps
import warnings

def lanczos(A, s=None, m=None):
    """
    Use m steps of the lanczos algorithm starting with q to generate
    the tridiagonal form of this matrix(The traditional scalar version).

    Parameters:
        :A: A sparse hermitian matrix.
        :s: The starting site, the 1-s site will be treated as a block.
        :m: The steps to run.
        :getbasis: Return basis vectors if True.

    Return:
        Tridiagonal part elements (data,offset),
        | data -> (lower part, middle part, upper part)
        | offset -> (-1, 0, 1) to indicate the value of (j-i) of specific data with i,j the matrix element indices.

    To construct the matrix, set the block-matrix elements with block indices j-i == offset[k] to data[k].
    This is exactly what `construct_tridmat` function do.

    **Note:** The initial vector q will be renormalized to guarant the correctness of result,
    """
    if sps.issparse(A): A=A.toarray()
    if m==None:
        m=A.shape[0]

    if s is None:
        s = 1

    #initialize states
    Q = np.zeros(A.shape[0],s)
    Q[:s,:s] = np.eye(s)
    
    #run steps
    for i in range(s,m):
        Q_=Q[:,i]
        z = A.dot(Q_)
        tmp = dot(conj(Q.T), z)
        tmp = dot(Q, tmp)
        z = z - tmp
        beta_i = sqrt(dot(conj(z),z))
        if i==m-1: break
        z=z/beta_i
        Q_i=icgs(z[:,newaxis],Q)
        Q=append(Q,Q_i,axis=-1)

    return Q

def icgs(u,Q,M=None,return_norm=False,maxiter=3):
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