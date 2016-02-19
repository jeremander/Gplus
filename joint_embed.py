"""Computes eigenvalues and eigenvectors of the joint PMI similarity matrices for three of the four attribute types. Rows are nodes, columns are features. Only keeps rows that have attributes."""

import pickle
import time
import numpy as np
import pandas as pd
import optparse
import sys
from scipy.sparse import coo_matrix, lil_matrix, diags
from collections import defaultdict
from gplus import *

attr_types = ['employer', 'major', 'places_lived', 'school']


def make_diag_block(attr_analyzer, attr_type, sim = 'NPMI1s', delta = 0.0):
    """Given an AttributeAnalyzer and an attribute type, returns the SparseLinearOperator for the diagonal block of the joint embedding. This is the "uncollapsed" PMI operator, where node rows are replicates of their corresponding attribute rows in the attribute PMI matrix."""
    n = attr_analyzer.num_vertices
    pfa = attr_analyzer.pairwise_freq_analyzers[attr_type]
    m = pfa.num_vocab
    attrs_by_node = attr_analyzer.attrs_by_node_by_type[attr_type]
    print("\nMaking PMI operator...")
    attr_block = pfa.to_sparse_PMI_operator(sim, delta)
    mapping = []
    for i in range(n):
        if i in attrs_by_node:
            mapping.append({pfa.vocab_indices[v] for v in attrs_by_node[i]})
        else:
            mapping.append({pfa.vocab_indices['*???*_%d' % i]})
    collapser = CollapseOperator(np.array(mapping), m)
    return SymmetricSparseLinearOperator(collapser.transpose() * attr_block * collapser)

def make_joint_embedding_operator(attr_analyzer, included_attr_types, sim = 'NPMI1s', delta = 0.0):
    """Makes the joint embedding operator for three of the four attribute types. This has three diagonal blocks, which are the uncollapsed PMI operators for each attribute type, and off-diagonal blocks corresponding to the means of these operators."""
    diag_blocks = []
    for attr_type in included_attr_types:
        print_flush("\nMaking %s block..." % attr_type)
        diag_blocks.append(make_diag_block(attr_analyzer, attr_type, sim, delta))
    return JointSymmetricBlockOperator(diag_blocks)


def main():
    p = optparse.OptionParser()
    p.add_option('--attr_type', '-a', type = str, help = 'attribute type to exclude')
    p.add_option('-p', type = str, default = 'NPMI1s', help = 'PMI type (PMIs, NPMI1s, or NPMI2s)')
    p.add_option('-e', type = str, default = 'adj', help = 'embedding (adj, normlap, regnormlap)')
    p.add_option('-d', type = float, default = 0.0, help = 'smoothing parameter')
    p.add_option('-k', type = int, default = 200, help = 'number of eigenvalues')
    p.add_option('-t', type = float, default = None, help = 'tolerance for eigsh')
    p.add_option('-v', action = 'store_true', default = False, help = 'save scree plot')
    opts, args = p.parse_args()

    attr_type = opts.attr_type
    other_attr_types = [at for at in attr_types if (at != attr_type)]
    sim = opts.p
    embedding = opts.e
    assert (embedding in ['adj', 'normlap', 'regnormlap'])
    delta = opts.d
    k = opts.k
    tol = opts.t
    save_plot = opts.v
    topN = 50  # for the report
    assert (((sim == 'PMIs') or (delta == 0)) and (sim in ['PMIs', 'NPMI1s', 'NPMI2s']))

    data_folder = 'gplus0_lcc/data/PMI/joint/'
    plot_folder = 'gplus0_lcc/plots/PMI/joint/'
    file_prefix1 = ('%s_%s_%s_delta' % (attr_type, sim, embedding)) + str(delta) + ('_k%d' % k)

    try:
        print_flush("\nLoading eigenvalues from '%s%s_eigvals.csv'..." % (data_folder, file_prefix1))
        eigvals = np.loadtxt('%s%s_eigvals.csv' % (data_folder, file_prefix1), delimiter = ',')
    except FileNotFoundError:
        print_flush("Failed to load.")
        print_flush("\nLoading AttributeAnalyzer...")
        a = AttributeAnalyzer(load_data = True)
        for at in other_attr_types:
            a.load_pairwise_freq_analyzer(at)
        a.make_attrs_by_node_by_type()
        n = sum([a.pairwise_freq_analyzers[at].num_vocab for at in other_attr_types])
        tol = (1.0 / n) if (tol is None) else tol  # use 1/n instead of machine precision as default tolerance
        print_flush("\nComputing joint similarity operator (%s)..." % sim)
        joint_op = make_joint_embedding_operator(a, other_attr_types, sim = sim, delta = delta)
        matrix_type = 'adjacency' if (embedding == 'adj') else ('normalized Laplacian' if (embedding == 'normlap') else 'regularized normalized Laplacian')
        print_flush("\nComputing eigenvectors of %s matrix (k = %d)..." % (matrix_type, k))
        if (embedding == 'adj'):
            (eigvals, features) = timeit(eigsh)(joint_op, k = k, tol = tol)
            features = np.sqrt(np.abs(eigvals)) * features  # scale the feature columns by the sqrt of the eigenvalues
        elif (embedding == 'normlap'):
            normlap = SparseNormalizedLaplacian(joint_op)
            (eigvals, features) = timeit(eigsh)(normlap, k = k, tol = tol)
        elif (embedding == 'regnormlap'):
            regnormlap = SparseRegularizedNormalizedLaplacian(joint_op)
            (eigvals, features) = timeit(eigsh)(regnormlap, k = k, tol = tol)

        num_attr_types = len(other_attr_types)
        features2 = np.zeros((len(a.attributed_nodes), num_attr_types * k))
        for i in range(num_attr_types):
            features2[:, i * k : (i + 1) * k] = features[i * a.num_vertices + np.array(a.attributed_nodes), :]

        np.savetxt('%s%s_eigvals.csv' % (data_folder, file_prefix1), eigvals, delimiter = ',')
        pickle.dump(features2, open('%s%s_joint_features.pickle' % (data_folder, file_prefix1), 'wb'))

    if save_plot:
        print_flush("\nSaving scree plot to '%s%s_screeplot.png'..." % (plot_folder, file_prefix1))
        scree_plot(eigvals, show = False, filename = '%s%s_screeplot.png' % (plot_folder, file_prefix1))

    print_flush("\nDone!")


if __name__ == "__main__":
    main()