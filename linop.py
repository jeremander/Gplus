from scipy.sparse.linalg import LinearOperator

class SparseLinearOperator(LinearOperator):
    """Subclass of LinearOperator for handling a sparse matrix."""
    def __init__(self, F):
        self.F = F
        super().__init__(dtype = float, shape = F.shape)
    def _matvec(self, x):
        return self.F * x
    def __getnewargs__(self):  # for pickling
        return (self.F,)


class SymmetricSparseLinearOperator(SparseLinearOperator):
    """Linear operator whose adjoint operator is the same, due to symmetry."""
    def _adjoint(self):
        return self


class PMILinearOperator(LinearOperator):
    """Subclass of LinearOperator for handling the sparse + low-rank PMI matrix. In particular, it represents the matrix F + Delta * 1 * 1^T - u * 1^T - 1 * u^T."""
    def __init__(self, F, Delta, u):
        n = F.shape[0]
        assert ((F.shape[1] == n) and (u.shape == (n,)))
        self.F, self.Delta, self.u = F, Delta, u
        self.u_prime = self.Delta - self.u
        self.ones = np.ones(n, dtype = float)
        super().__init__(dtype = float, shape = self.F.shape)
    def _matvec(self, x):
        return self.F * x + self.u_prime * np.sum(x) - self.ones * np.dot(self.u, x)
    def _adjoint(self):
        return self
    def __getnewargs__(self):  # for pickling
        return (self.F, self.Delta, self.u)