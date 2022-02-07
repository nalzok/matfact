import numpy as np
import scipy.linalg as la


# TODO:
# [X] Debug issues above
# [ ] Rational arithmetic (try Python libaries first),
# [ ] Shape commutation algorithm (Prop 7)
# [ ] Quiver algorithm (Alg 5)
# 
# Last resort:
# https://caam37830.github.io/book/00_python/classes.html


def leup(M: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    >>> m, n = 5, 4
    >>>
    >>>
    >>> M = np.random.randn(m, n)
    >>> L, E, U, P = leup(M)
    >>> np.allclose(M, L @ E @ U @ P)
    True
    >>> is_EL(E)
    True
    >>>
    >>> M = np.random.randn(m, n)
    >>> M[0, 0] = 0
    >>> L, E, U, P = leup(M)
    >>> np.allclose(M, L @ E @ U @ P)
    True
    >>> is_EL(E)
    True
    >>>
    >>> M = np.random.randn(m, n)
    >>> M[0] = 0
    >>> L, E, U, P = leup(M)
    >>> np.allclose(M, L @ E @ U @ P)
    True
    >>> is_EL(E)
    True
    >>>
    >>> M = np.random.randn(m, n)
    >>> M[:, 0] = 0
    >>> L, E, U, P = leup(M)
    >>> np.allclose(M, L @ E @ U @ P)
    True
    >>> is_EL(E)
    True
    """
    m, n = M.shape
    E = np.array(M, copy=True)
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
        
        # We don't need to form the elimination matrices explictly
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
    doctest.testmod()
