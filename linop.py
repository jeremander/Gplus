import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg.interface import _ProductLinearOperator
from scipy.sparse import csr_matrix, lil_matrix
from functools import reduce

class ProductLinearOperator(_ProductLinearOperator):
    """Subclass of scipy's ProductLinearOperator that can suppport pickling."""
    def __init__(self, A, B):
        super().__init__(A, B)
    def __getnewargs__(self):
        return self.args

class SparseLinearOperator(LinearOperator):
    """Subclass of LinearOperator for handling a sparse matrix."""
    def __init__(self, F):
        """Input must be a sparse matrix or SparseLinearOperator."""
        self.F = F
        #super().__init__(matvec = lambda x : SparseLinearOperator._matvec(self, x), dtype = float, shape = F.shape)
        super().__init__(dtype = float, shape = F.shape)
    def get(self, i, j):
        """Gets element i,j from the matrix representation of the SparseLinearOperator."""
        vi, vj = np.zeros(self.shape[0], dtype = int), np.zeros(self.shape[1], dtype = int)
        vi[i] = 1
        vj[j] = 1
        return np.dot(vi, self * vj)
    def todense(self):
        """Returns as a dense matrix. Warning: do not use this if the dimensions are large."""
        result = np.zeros(self.shape, dtype = float)
        for j in range(self.shape[1]):
            v = np.zeros(self.shape[1], dtype = float)
            v[j] = 1.0
            result[:, j] = self._matvec(v)
        return result
    def _matvec(self, x):
        return self.F * x
    def _transpose(self):
        return SparseLinearOperator(self.F.transpose())
    def __getnewargs__(self):  # for pickling
        return (self.F,)

class SymmetricSparseLinearOperator(SparseLinearOperator):
    """Linear operator whose adjoint operator is the same, due to symmetry."""
    def _adjoint(self):
        return self
    def _transpose(self):
        return self

class DiagonalLinearOperator(SymmetricSparseLinearOperator):
    """Linear operator representing a diagonal matrix."""
    def __init__(self, D):
        """D is a 1D array containing the diagonal entries."""
        self.D = D
        LinearOperator.__init__(self, dtype = float, shape = (len(D), len(D)))
    def _matvec(self, x):
        return self.D * x
    def __getnewargs__(self):
        return (self.D,)

class ConstantDiagonalLinearOperator(DiagonalLinearOperator):
    """Linear operator representing a constant diagonal matrix."""
    def __init__(self, n, c):
        """n is dimension, c is a constant to be multiplied by the identity matrix."""
        self.c = c
        LinearOperator.__init__(self, dtype = float, shape = (n, n))
    def _matvec(self, x):
        return self.c * x
    def __getnewargs__(self):
        return (self.c,)

class PMILinearOperator(SymmetricSparseLinearOperator):
    """Subclass of LinearOperator for handling the sparse + low-rank PMI matrix. In particular, it represents the matrix F + Delta * 1 * 1^T - u * 1^T - 1 * u^T."""
    def __init__(self, F, Delta, u):
        n = F.shape[0]
        assert ((F.shape[1] == n) and (u.shape == (n,)))
        self.F, self.Delta, self.u = F, Delta, u
        self.u_prime = self.Delta - self.u
        self.ones = np.ones(n, dtype = float)
        LinearOperator.__init__(self, dtype = float, shape = self.F.shape)
    def _matvec(self, x):
        return self.F * x + self.u_prime * np.sum(x) - self.ones * np.dot(self.u, x)
    def __getnewargs__(self):  # for pickling
        return (self.F, self.Delta, self.u)

class SparseLaplacian(SymmetricSparseLinearOperator):
    """Class for representing a sparse Laplacian (D - A). Can also subclass the normalized version (D^(-1/2) * A * D^(-1/2)) or the regularized normalized version ((D + tau * I)^(-1/2) * A * (D + tau * I)^(-1/2)). Constructs the Laplacian from a SymmetricSparseLinearOperator representing the adjacency matrix."""
    def __init__(self, A):
        assert isinstance(A, SymmetricSparseLinearOperator)
        self.A = A
        self.D = self.A.matvec(np.ones(self.A.shape[1], dtype = float))
        LinearOperator.__init__(self, dtype = float, shape = A.shape)
    def _matvec(self, x):
        return (self.D * x - self.A.matvec(x))

class SparseNormalizedLaplacian(SparseLaplacian):
    """Class representing the normalized Laplacian (D^(-1/2) * A * D^(-1/2))."""
    def __init__(self, A):
        super().__init__(A)
        self.D_inv_sqrt = 1.0 / np.sqrt(self.D)  # hopefully D has all positive entries
    def _matvec(self, x):
        return (x - self.D_inv_sqrt * self.A._matvec(self.D_inv_sqrt * x))

class SparseRegularizedNormalizedLaplacian(SparseLaplacian):
    """Class representing the regularized normalized Laplacian ((D + tau * I)^(-1/2) * A * (D + tau * I)^(-1/2))."""
    def __init__(self, A):
        super().__init__(A)
        tau = np.mean(self.D)
        self.D_plus_tau_I_inv_sqrt = 1.0 / np.sqrt(self.D + tau)
    def _matvec(self, x):
        return (x - self.D_plus_tau_I_inv_sqrt * self.A._matvec(self.D_plus_tau_I_inv_sqrt * x))

class SparseDiagonalAddedAdjacencyOperator(SymmetricSparseLinearOperator):
    """Class representing an adjacency matrix A + D/n."""
    def __init__(self, A):
        assert isinstance(A, SymmetricSparseLinearOperator)
        self.A = A
        LinearOperator.__init__(self, dtype = float, shape = A.shape)
        self.D_ratio = self.A._matvec(np.ones(self.A.shape[1], dtype = float)) / self.shape[0]
    def _matvec(self, x):
        return (self.A.matvec(x) + self.D_ratio * x)

class BlockSparseLinearOperator(SparseLinearOperator):
    """Class representing a block structure of sparse linear operators."""
    def __init__(self, block_grid):
        """Input is a 2D list of SparseLinearOperators. The resulting operator is the corresponding operator comprised of these operator blocks. The dimensions must match correctly. This assumes the number of blocks in each row and column is the same."""
        self.block_grid_shape = (len(block_grid), len(block_grid[0]))
        # validate block dimensions
        assert all([len(row) == self.block_grid_shape[1] for row in block_grid]), "Must be same number of blocks in each row."
        assert all([len(set([block_grid[i][j].shape[0] for j in range(self.block_grid_shape[1])])) == 1 for i in range(self.block_grid_shape[0])]), "dimension mismatch"
        assert all([len(set([block_grid[i][j].shape[1] for i in range(self.block_grid_shape[0])])) == 1 for j in range(self.block_grid_shape[1])]), "dimension mismatch"
        shape = (sum([block_grid[i][0].shape[0] for i in range(len(block_grid))]), sum([block_grid[0][j].shape[1] for j in range(len(block_grid[0]))]))
        # compute transition indices between blocks
        self.row_indices = [0] + list(np.cumsum([row[0].shape[0] for row in block_grid]))
        self.column_indices = [0] + list(np.cumsum([block.shape[1] for block in block_grid[0]]))
        self.block_grid = block_grid
        LinearOperator.__init__(self, dtype = float, shape = shape)
    def _matvec(self, x):
        result = np.zeros(self.shape[0], dtype = float)
        for i in range(self.block_grid_shape[0]):
            row = self.block_grid[i]
            partial_result = np.zeros(row[0].shape[0], dtype = float)
            for j in range(self.block_grid_shape[1]):
                partial_result += row[j]._matvec(x[self.column_indices[j] : self.column_indices[j + 1]])
            result[self.row_indices[i] : self.row_indices[i + 1]] = partial_result
        return result
    def __getnewargs__(self):  # for pickling
        return (self.block_grid,)

class CollapseOperator(SparseLinearOperator):
    """Given a mapping from [0 ... (n - 1)] to Pow([0 ... (m - 1)]), with m <= n, represents the m x n linear operator that sums together vector entries belonging to the same equivalence class under this mapping."""
    def __init__(self, mapping, m):
        """mapping is an n-long array of sets of integers in [0 ... (m - 1)]."""
        self.mapping = mapping
        self.m = m
        n = len(mapping)
        assert (m <= n)
        assert (m > max((reduce(max, s, -1) for s in mapping)))
        assert (min(map(len, mapping)) > 0)
        mat = lil_matrix((m, n), dtype = float)
        for (i, img) in enumerate(mapping):
            for j in img:
                mat[j, i] = 1.0 / len(img)
        SparseLinearOperator.__init__(self, mat.tocsr())
    def __getnewargs__(self):  # for pickling
        return (self.mapping, self.m)

class JointSymmetricBlockOperator(SymmetricSparseLinearOperator):
    """Given a list of SymmetricSparseLinearOperators of the same dimension, constructs a joint operator where the diagonal blocks are the input operators, and the off-diagonal blocks are the means of these operators."""
    def __init__(self, diag_blocks, tau = None):
        self.diag_blocks = diag_blocks
        self.tau = tau
        self.num_blocks = len(diag_blocks)
        assert (isinstance(diag_blocks, list) and len(diag_blocks) > 0)
        assert all([isinstance(block, SymmetricSparseLinearOperator) for block in diag_blocks]), "Blocks must be symmetric."
        assert all([block.shape == diag_blocks[0].shape for block in diag_blocks]), "Blocks must have the same shape."
        self.n = diag_blocks[0].shape[0]
        if (tau is None):
            joint_block_operator = BlockSparseLinearOperator([[diag_blocks[i] if (i == j) else (0.5 * np.sum([diag_blocks[i], diag_blocks[j]])) for j in range(self.num_blocks)] for i in range(self.num_blocks)])
        else:
            scaled_identity = ConstantDiagonalLinearOperator(diag_blocks[0].shape[0], tau)
            joint_block_operator = BlockSparseLinearOperator([[diag_blocks[i] if (i == j) else (0.5 * np.sum([diag_blocks[i], diag_blocks[j]]) + scaled_identity) for j in range(self.num_blocks)] for i in range(self.num_blocks)])
        SymmetricSparseLinearOperator.__init__(self, joint_block_operator)
        def __getnewargs__(self):
            return (self.diag_blocks, self.tau)




# test1 = SymmetricSparseLinearOperator(csr_matrix(np.array([[1.,2.],[2.,3.]])))
# test2 = SparseLinearOperator(csr_matrix(np.array([[1.,0,0,0],[0,1,0,0]])))
# zeros = SymmetricSparseLinearOperator(csr_matrix(np.zeros((4, 4), dtype = float)))
# test3 = BlockSparseLinearOperator([[test1, test2], [test2.transpose(), zeros]])
# mapping1 = np.array([{0}, {1}, {1}, {2}, {2}, {2}])
# mapping2 = np.array([{0}, {0, 1}, {1}, {1, 2}, {2}, {2}])
# x = np.array([-5., 3., 2., 1., 1., 5.])
# A = SymmetricSparseLinearOperator(csr_matrix(np.array([[1.0, 0.3, 0.7], [0.3, 1.0, 0.1], [0.7, 0.1, 1.0]])))

