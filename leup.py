import numpy as np
from scipy import linalg as la


def leup(M: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Perform LEUP factorization described by algo 4 of <arXiv:1911.10693>.

    >>> np.random.seed(42)
    >>>
    >>> m, n = 5, 4
    >>> M1 = np.random.randn(m, n)
    >>> M2 = np.random.randn(m, n)
    >>> M2[0, 0] = 0
    >>> M3 = np.random.randn(m, n)
    >>> M3[0] = 0
    >>> M4 = np.random.randn(m, n)
    >>> M4[:, 0] = 0
    >>> M5 = sparse.random(5, 4, density=0.2).toarray().astype(np.float64)
    >>>
    >>> for M in [M1, M2, M3, M4, M5]:
    ...     L, E, U, P = leup(M)
    ...     assert np.allclose(M, L @ E @ U @ P)
    ...     assert is_EL(E)
    """
    m, n = M.shape
    E = M.copy()
    L = np.eye(m)
    U = np.eye(n)
    P = np.eye(n)

    i, j = 0, 0
    while i < m and j < n:
        # find column to swap with
        for j2 in range(j, n):
            if E[i, j2] != 0:
                break
        else:
            # increment row
            i += 1
            continue

        # else, we have a pivot
        # swap columns of E, U
        E[i:, [j, j2]] = E[i:, [j2, j]]
        U[:, [j, j2]] = U[:, [j2, j]]
        
        # swap rows of U and P
        U[[j, j2]] = U[[j2, j]] 
        P[[j, j2]] = P[[j2, j]] 

        # loop invariant
        assert np.allclose(M, L @ E @ U @ P), f'Swap failure at {i=}, {j=}'
        
        # now do a schur complement
        Aij = E[i:i+1, j:j+1]
        A2j = E[i+1:, j:j+1]
        Ai2 = E[i:i+1, j+1:]
        A22 = E[i+1:, j+1:]
        
        # no need to form the elimination matrices explicitly
        L[i+1:, i:i+1] = la.solve(Aij.T, A2j.T).T
        U[j:j+1, j+1:] = la.solve(Aij, Ai2)

        # form schur complement and apply
        S = A22 - A2j @ la.solve(Aij, Ai2)
        E[i+1:, j+1:] = S
        E[i:i+1, j+1:] = 0
        E[i+1:, j:j+1] = 0

        # loop invariant
        assert np.allclose(M, L @ E @ U @ P), f'Schur failure at {i=}, {j=}'

        j += 1
        i += 1
        
    return L, E, U, P
        

def is_EL(A: np.ndarray) -> bool:
    """
    Check if a matrix is an echelon pivot matrix

    >>> A = np.array([
    ...     [1, 0, 0],
    ...     [0, 0, 0],
    ...     [0, 1, 0],
    ... ])
    >>> is_EL(A)
    True
    >>> A = np.array([
    ...     [1, 0, 0],
    ...     [0, 1, 0],
    ...     [0, 1, 0],
    ... ])
    >>> is_EL(A)
    False
    >>> A = np.array([
    ...     [1, 0, 0],
    ...     [0, 1, 1],
    ...     [0, 0, 0],
    ... ])
    >>> is_EL(A)
    False
    """
    nz = np.transpose(np.nonzero(A))
    nz = nz[nz[:, 1].argsort()]
    for (i1, j1), (i2, j2) in zip(nz, nz[1:]):
        if not (i2 > i1 and j2 == j1 + 1):
            return False

    return True


if __name__ == "__main__":
    import doctest
    from scipy import sparse
    doctest.testmod()

