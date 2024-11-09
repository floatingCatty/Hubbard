import torch
from typing import List
import math

def sparse_kron(input: torch.Tensor, other: torch.Tensor):
    """
    https://github.com/pytorch/pytorch/issues/134069
    """
    assert input.ndim == other.ndim
    input_indices = input.indices()
    other_indices = other.indices()

    input_indices_expanded = input_indices.expand(other_indices.shape[1], *input_indices.shape).T * torch.tensor(other.shape).reshape(1,-1,1)
    other_indices_expanded = other_indices.expand(input_indices.shape[1], *other_indices.shape)
    new_indices = torch.permute(input_indices_expanded + other_indices_expanded, (1,0,2)).reshape(input.ndim,-1)

    new_values = torch.kron(input.values(), other.values())

    if new_indices.ndim == 1:
        new_indices = new_indices.reshape([input.ndim, 0])

    new_shape = [n * m for n, m in zip(input.shape, other.shape)]

    return torch.sparse_coo_tensor(new_indices, new_values, new_shape, dtype=input.dtype, device=input.device)


def ksum(matrices: List[torch.Tensor], coeff: torch.Tensor=None):
    """
    Compute the Kronecker sum of a list of square matrices.

    Parameters:
    matrices (list of torch.Tensor): List of square matrices [A, B, C, ...].

    Returns:
    torch.Tensor: The Kronecker sum matrix.
    """
    if coeff is not None:
        assert len(coeff) == len(matrices), "The number of coefficients should match the number of matrices."
        matrices = [matrices[i] * coeff[i] for i in range(len(matrices))]

    K = matrices[0]
    for A_i in matrices[1:]:
        n1 = K.size(0)
        n2 = A_i.size(0)
        # Compute Kronecker products efficiently without forming large identity matrices
        K = torch.kron(K, torch.eye(n2)) + torch.kron(torch.eye(n1), A_i)
    return K

def kdsum(matricesA: List[torch.Tensor], matricesB: List[torch.Tensor], coeff: torch.Tensor=None):
    """
    Compute the sum over all pairs (i != j) of Kronecker products where
    each term has identities except at positions i and j, where we have A_i and A_j.

    result = \sum_{i!=j}C_{ij} I_1 ⊗ ... ⊗ A_i(.H) ⊗ ... ⊗ A_j ⊗ ... ⊗ I_n

    Parameters:
    matrices (list of torch.Tensor): List of square matrices [A_1, A_2, ..., A_n].

    Returns:
    torch.Tensor: The resulting matrix of the Kronecker sum.
    """
    if coeff is not None:
        assert coeff.shape[0] == len(matricesA), "The number of coefficients should match the number of matrices."
        assert coeff.shape[1] == len(matricesA), "The number of coefficients should match the number of matrices."
        assert coeff.shape[0] == len(matricesB), "The number of coefficients should match the number of matrices."
        assert coeff.shape[1] == len(matricesB), "The number of coefficients should match the number of matrices."

    n = len(matricesA)
    sizes = [A.size(0) for A in matricesA]
    total_size = torch.prod(torch.tensor(sizes)).item()

    # Initialize result matrix
    result = torch.zeros((total_size, total_size), dtype=matricesA[0].dtype, device=matricesA[0].device)

    # Generate all pairs of indices (i, j) where i != j
    index_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    # For each pair (i, j), compute the Kronecker product and add to result
    for i, j in index_pairs:
        # Build list of matrices for Kronecker product
        kron_matrices = []
        for k in range(n):
            if k == i:
                kron_matrices.append(matricesA[k])
            elif k == j:
                kron_matrices.append(matricesB[k])
            else:
                # Use identity matrices without explicitly creating them
                kron_matrices.append(None)  # Placeholder for identity

        # Compute Kronecker product efficiently
        # Start with the first non-identity matrix
        kron_product = None
        for idx, mat in enumerate(kron_matrices):
            if mat is not None:
                if kron_product is None:
                    kron_product = mat
                else:
                    kron_product = torch.kron(kron_product, mat)
            else:
                # Update kron_product to include the size of the identity matrix
                identity_size = sizes[idx]
                if kron_product is None:
                    kron_product = torch.eye(identity_size, dtype=matricesA[0].dtype, device=matricesA[0].device)
                else:
                    kron_product = torch.kron(kron_product, torch.eye(identity_size, dtype=matricesA[0].dtype, device=matricesA[0].device))

        # Add the computed Kronecker product to the result
        if coeff is not None:
            kron_product = coeff[i, j] * kron_product
        result += kron_product

    return result

def ksumact(matrices: List[torch.Tensor], x: torch.Tensor, coeff: torch.Tensor=None):
    """
    Compute the action of the Kronecker sum on a vector.

    Parameters:
    matrices (list of torch.Tensor): List of square matrices [A, B, C, ...].
    x (torch.Tensor): Input vector. Shape [..., A.size(0) * B.size(0) * C.size(0) * ...]

    Returns:
    torch.Tensor: Resulting vector after applying the Kronecker sum.
    """

    if coeff is not None:
        assert len(coeff) == len(matrices), "The number of coefficients should match the number of matrices."
        matrices = [matrices[i] * coeff[i] for i in range(len(matrices))]

    sizes = [A.size(0) for A in matrices]
    assert x.shape[-1] == math.prod(sizes), "Input vector has incorrect size."
    orgx_shape = list(x.shape)
    pre_dim = len(orgx_shape) - 1
    X = x.reshape(orgx_shape[:-1]+sizes)
    Y = torch.zeros_like(X)
    for i, A_i in enumerate(matrices):
        # Permute dimensions to bring the i-th axis to the front
        permute_order = [pre_dim+i] + list(range(0, pre_dim+i)) + list(range(pre_dim+i+1, pre_dim+len(sizes)))
        X_perm = X.permute(permute_order)
        # Reshape for matrix multiplication
        X_flat = X_perm.reshape(sizes[i], -1)
        # Apply the matrix A_i
        Y_flat = A_i @ X_flat
        # Reshape back to original tensor shape
        Y_i = Y_flat.reshape(X_perm.shape)
        # Permute dimensions back to original order
        permute_back = list(range(1, pre_dim+i+1)) + [0] + list(range(pre_dim+i+1, pre_dim+len(sizes)))
        Y_i = Y_i.permute(permute_back)
        # Accumulate the result
        Y += Y_i
    return Y.reshape(orgx_shape[:-1]+[-1])

def kdsumact(matricesA: List[torch.Tensor], matricesB: List[torch.Tensor], x: torch.Tensor, coeff: torch.Tensor=None):
    """
    Optimized function to compute the action of the custom operator on a vector x.
    
    The operator is defined as:
    result = sum_{i != j} (A_i ⊗ A_j) acting on x, with identities on other dimensions.
    
    Parameters:
    matrices (list of torch.Tensor): List of square matrices [A_1, A_2, ..., A_n].
    x (torch.Tensor): Input vector of appropriate size. x (torch.Tensor): Input vector. 
        Shape [..., A.size(0) * B.size(0) * C.size(0) * ...]
    
    Returns:
    torch.Tensor: Resulting vector after applying the operator.
    """

    if coeff is not None:
        assert coeff.shape[0] == len(matricesA), "The number of coefficients should match the number of matrices."
        assert coeff.shape[1] == len(matricesB), "The number of coefficients should match the number of matrices."

    n = len(matricesA)
    sizes = [A.size(0) for A in matricesA]
    total_size = torch.prod(torch.tensor(sizes)).item()
    
    org_shape = list(x.shape)
    pre_dim = len(org_shape) - 1
    assert org_shape[-1] == total_size, "Input vector has incorrect size."
    
    # Reshape x into a tensor of shape [n1, n2, ..., nn]
    X = x.reshape(org_shape[:-1]+sizes)
    Y = torch.zeros_like(X)
    
    # Precompute the application of each A_i along its axis
    applied_matricesA = []
    applied_matricesB = []
    for idx, A in enumerate(matricesA):
        # Apply A along its axis
        X_perm = X.movedim(pre_dim+idx, 0)
        X_flat = X_perm.reshape(sizes[idx], -1)
        Y_flat = A @ X_flat
        Y_perm = Y_flat.reshape(X_perm.shape)
        Y_back = Y_perm.movedim(0, pre_dim+idx)
        applied_matricesA.append(Y_back)

        Y_flat = matricesB[idx] @ X_flat
        Y_perm = Y_flat.reshape(X_perm.shape)
        Y_back = Y_perm.movedim(0, pre_dim+idx)
        applied_matricesB.append(Y_back)
    
    # Sum over i != j using vectorized operations
    for i in range(n):
        for j in range(i + 1, n):
            # Compute the double application without explicit loops
            temp = applied_matricesA[i]
            temp_perm = temp.movedim(pre_dim+j, 0)
            temp_flat = temp_perm.reshape(sizes[j], -1)
            temp_result = matricesB[j] @ temp_flat
            temp_result = temp_result.reshape(temp_perm.shape)
            temp_result = temp_result.movedim(0, pre_dim+j)
            if coeff is not None:
                temp_result = coeff[i,j] * temp_result
            Y += temp_result
            
            # Symmetric term (since i != j)
            temp = applied_matricesA[j]
            temp_perm = temp.movedim(pre_dim+i, 0)
            temp_flat = temp_perm.reshape(sizes[i], -1)
            temp_result = matricesB[i] @ temp_flat
            temp_result = temp_result.reshape(temp_perm.shape)
            temp_result = temp_result.movedim(0, pre_dim+i)
            if coeff is not None:
                temp_result = coeff[j,i] * temp_result
            Y += temp_result
    
    return Y.reshape(org_shape[:-1]+[-1])


if __name__ == "__main__":
    import time
    torch.set_default_dtype(torch.float64)
    # test kronecker_sum
    A = torch.randn(20,20)
    B = torch.randn(4,4)
    C = torch.randn(12,12)
    vec = torch.randn(10,A.shape[0]*B.shape[0]*C.shape[0])


    start = time.time()
    D = ksum([A, B, C])
    end = time.time()

    D_ = torch.kron(torch.kron(A, torch.eye(B.shape[0])), torch.eye(C.shape[0])) + torch.kron(torch.kron(torch.eye(A.shape[0]), B), torch.eye(C.shape[0])) + torch.kron(torch.kron(torch.eye(A.shape[0]), torch.eye(B.shape[0])), C)

    print("error kronsum:\t", (D-D_).abs().sum().item(), "\ttime:\t", end-start)

    # test kronsum_action
    start = time.time()
    avec = ksumact([A, B, C], vec)
    end = time.time()
    print("error kronsum_action:\t", ((D_ @ vec.T).T - avec).abs().sum().item(), "\ttime:\t", end-start)

    # test krondsum
    start = time.time()
    D = kdsum([A, B, C], [A, B, C])
    end = time.time()

    D_ = torch.zeros_like(D)
    D_ += 2 * torch.kron(torch.kron(A, B), torch.eye(C.shape[0])) + 2 * torch.kron(torch.kron(A, torch.eye(B.shape[0])), C) + 2 * torch.kron(torch.kron(torch.eye(A.shape[0]), B), C)

    print("error krondsum:\t", (D-D_).abs().sum().item(), "\ttime:\t", end-start)

    # test krondsum_action
    start = time.time()
    avec = kdsumact([A, B, C], [A, B, C], vec)
    end = time.time()
    print("error krondsum_action\t", ((D_ @ vec.T).T - avec).abs().sum().item(), "\ttime:\t", end-start)