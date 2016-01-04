import numpy as np
from scipy.sparse.linalg import LinearOperator

class SparseLinearOperator(LinearOperator):
    """Subclass of LinearOperator for handling a sparse matrix."""
    def __init__(self, F):
        self.F = F
        #super().__init__(matvec = lambda x : SparseLinearOperator._matvec(self, x), dtype = float, shape = F.shape)
        super().__init__(dtype = float, shape = F.shape)
    def get(self, i, j):
        """Gets element i,j from the matrix representation of the SparseLinearOperator."""
        vi, vj = np.zeros(self.shape[0], dtype = int), np.zeros(self.shape[1], dtype = int)
        vi[i] = 1
        vj[j] = 1
        return np.dot(vi, self * vj)
    def _matvec(self, x):
        return self.F * x
    def __getnewargs__(self):  # for pickling
        return (self.F,)


class SymmetricSparseLinearOperator(SparseLinearOperator):
    """Linear operator whose adjoint operator is the same, due to symmetry."""
    def _adjoint(self):
        return self


class DiagonalLinearOperator(SymmetricSparseLinearOperator):
    """Linear operator representing a diagonal matrix."""
    def __init__(self, D):
        """D is a 1D array containing the diagonal entries."""
        self.D = D
        LinearOperator.__init__(self, dtype = float, shape = (len(D), len(D)))
    def _matvec(self, x):
        return self.D * x


class PMILinearOperator(SymmetricSparseLinearOperator):
    """Subclass of LinearOperator for handling the sparse + low-rank PMI matrix. In particular, it represents the matrix F + Delta * 1 * 1^T - u * 1^T - 1 * u^T."""
    def __init__(self, F, Delta, u):
        n = F.shape[0]
        assert ((F.shape[1] == n) and (u.shape == (n,)))
        self.F, self.Delta, self.u = F, Delta, u
        self.u_prime = self.Delta - self.u
        self.ones = np.ones(n, dtype = float)
        #LinearOperator.__init__(self, matvec = lambda x : PMILinearOperator._matvec(self, x), dtype = float, shape = F.shape)
        LinearOperator.__init__(self, dtype = float, shape = self.F.shape)
    def _matvec(self, x):
        return self.F * x + self.u_prime * np.sum(x) - self.ones * np.dot(self.u, x)
    def __getnewargs__(self):  # for pickling
        return (self.F, self.Delta, self.u)


class SparseLaplacian(SymmetricSparseLinearOperator):
    """Class for representing a sparse Laplacian. Constructs either the normalized or un-normalized version from a SymmetricSparseLinearOperator representing the adjacency matrix."""
    def __init__(self, A, normalized = False):
        assert isinstance(A, SymmetricSparseLinearOperator)
        self.A = A
        self.normalized = normalized
        self.D = self.A.matvec(np.ones(self.A.shape[1], dtype = float))
        self.D_inv_sqrt = 1.0 / np.sqrt(self.D)  # hopefully D has all positive entries
        LinearOperator.__init__(self, dtype = float, shape = A.shape)
    def _matvec(self, x):
        if self.normalized:
            return (x - self.D_inv_sqrt * self.A.matvec(self.D_inv_sqrt * x))
        else:
            return (self.D * x - self.A.matvec(x))

