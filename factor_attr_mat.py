"""Computes eigenvalues and eigenvectors of the NPMI1 sparse matrices for a given attribute type."""

import pickle
import time
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh


def main():
    p = optparse.OptionParser()
    p.add_option('--attr_type', '-a', type = str, help = 'attribute type')
    p.add_option('-k', type = int, help = 'number of eigenvalues')
    opts, args = p.parse_args()

    df = pd.read_csv("gplus0_lcc/data/npmi1s_%s.dat" % opts.attr_type, header = None, sep = ' ')
    df.columns = ['row', 'col', 'data']
    n = max(max(df['row']), max(df['col']))
    mat = coo_matrix((df['data'], (df['row'], df['col'])), shape = (n, n)).tocsr()
    mat = mat + mat.transpose().tocsr() - diags(mat.diagonal(), offsets = 0).tocsr()  # symmetrize
    start_time = time.time()
    (eigenvals, eigenvecs) = eigsh(mat, k = opts.k)
    elapsed_time = time.time() - start_time
    print("Time = %.3f seconds" % elapsed_time)
    pickle.dump((eigenvals, eigenvecs), open("gplus0_lcc/data/npmi1s_%s_eigendecomp.pickle", 'wb'))


if __name__ == "__main__":
    main()