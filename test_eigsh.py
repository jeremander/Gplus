from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from itertools import product
import numpy as np
import optparse
import time

def random_sparse_sym_matrix(n, p):
    """Constructs a sparse random nxn matrix with approximately p * n^2 uniformly distributed entries."""
    nnz = int(p * n * (n + 1) / 2)
    entries = dict()
    ctr = 0
    while (ctr < nnz):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        pair = tuple(sorted([i, j]))
        if (pair in entries):
            continue
        entries[pair] = np.random.rand()
        ctr += 1
    rows, cols, data = np.zeros(nnz, dtype = int), np.zeros(nnz, dtype = int), np.zeros(nnz, dtype = float)
    for (k, ((i, j), val)) in enumerate(entries.items()):
        rows[k] = i
        cols[k] = j
        data[k] = val
    mat = coo_matrix((data, (rows, cols)), shape = (n, n)).tocsr()
    mat = mat + mat.transpose().tocsr() - diags(mat.diagonal(), offsets = 0).tocsr()  # symmetrize
    return mat


def main():
    p = optparse.OptionParser()
    p.add_option('-n', type = int, help = 'dimension of matrix')
    p.add_option('-p', type = float, help = 'probability nonzero')
    p.add_option('-k', type = int, help = 'number of eigenvalues')
    opts, args = p.parse_args()
    mat = random_sparse_sym_matrix(opts.n, opts.p)
    print(repr(mat))
    start_time = time.time()
    result = eigsh(mat, k = opts.k)
    elapsed_time = time.time() - start_time
    print("Time = %.3f seconds" % elapsed_time)

if __name__ == "__main__":
    main()

