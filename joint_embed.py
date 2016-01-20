"""Computes eigenvalues and eigenvectors of the joint PMI similarity matrices for three of the four attribute types. Saves the results of this along with kMeans clustering of the attributes."""

import pickle
import time
import numpy as np
import pandas as pd
import optparse
import sys
from scipy.sparse import coo_matrix, lil_matrix, diags
from sklearn.cluster import KMeans
from collections import defaultdict
from gplus import *

attr_types = ['employer', 'major', 'places_lived', 'school']


class AttrsToNodes(ObjectWithReadwriteProperties):
    """Makes mapping from attributes to nodes possessing the attributes, for each attribute type."""
    readwrite_properties = {'nodes_by_attr_by_type' : 'pickle'}
    def __init__(self, folder = 'gplus0_lcc/data'):
        ObjectWithReadwriteProperties.__init__(self, folder)
    @autoreadwrite(['nodes_by_attr_by_type'], ['pickle'])
    def make_nodes_by_attr_by_type(self, load = True, save = False):
        self._nodes_by_attr_by_type = dict((attr_type, defaultdict(set)) for attr_type in attr_types)
        node_attr_filename = self.folder + '/node_attributes.csv'
        attr_df = pd.read_csv(node_attr_filename, sep = ';')
        for (i, node, attr_type, attr) in attr_df.itertuples():
            self._nodes_by_attr_by_type[attr_type][str(attr)].add(node)
    def get_nodes(self, attr_type, attr):
        if attr.startswith('*???*_'):
            return {int(attr[6:])}
        return self._nodes_by_attr_by_type[attr_type][attr]

def make_block(attr_analyzer, attrs_to_nodes, attr_type1, attr_type2, sim = 'NPMI1s', delta = 0.0):
    """Given an AttributeAnalyzer and two attribute types, returns the SparseLinearOperator required for the corresponding block in the joint embedding. If the attribute types are the same, this is the inter-attribute PMI matrix. If they are different, it is the Jaccard similarity between attributes (equivalence classes of nodes possessing each attribute). If one of the attributes is an 'unknown', the similarity will be 1 or 0 depending on whether the node possesses the attribute; if both are 'unknown', the similarity will be 1 or 0 depending on whether the nodes are identical."""
    pfa1 = attr_analyzer.pairwise_freq_analyzers[attr_type1]
    pfa2 = attr_analyzer.pairwise_freq_analyzers[attr_type2]
    attrs_by_node2 = attr_analyzer.attrs_by_node_by_type[attr_type2]
    nodes_by_attr1 = attrs_to_nodes.nodes_by_attr_by_type[attr_type1]
    nodes_by_attr2 = attrs_to_nodes.nodes_by_attr_by_type[attr_type2]
    if (attr_type1 == attr_type2):
        block = pfa1.to_sparse_PMI_operator(sim, delta)
    else:
        block = lil_matrix((pfa1.num_vocab, pfa2.num_vocab), dtype = float)
        for (i, v1) in enumerate(pfa1.vocab):
            if v1.startswith('*???*_'):
                try:
                    j = pfa2.vocab_indices[v1]
                    block[i, j] = 1.0
                except KeyError:
                    attrs2 = attrs_by_node2[int(v1[6:])]
                    for v2 in attrs2:
                        block[i, pfa2.vocab_indices[v2]] = 1.0 / len(nodes_by_attr2[v2])
            else:  # for efficiency, get all attributes2 sharing any node with attribute1, then compute Jaccard similarity for each
                connected_attrs2 = set()
                nodes1 = nodes_by_attr1[v1]
                for node in nodes1:
                    connected_attrs2.update(attrs_by_node2[node])
                for v2 in connected_attrs2:
                    nodes2 = nodes_by_attr2[v2]
                    block[i, pfa2.vocab_indices[v2]] = len(nodes1.intersection(nodes2)) / float(len(nodes1.union(nodes2)))
    return SparseLinearOperator(block.tocsr())

def make_joint_embedding_operator(attr_analyzer, attrs_to_nodes, included_attr_types, sim = 'NPMI1s', delta = 0.0):
    """Makes the joint embedding operator for three of the four attribute types. This has three diagonal blocks corresponding to the PMI operators for each attribute type, and off-diagonal blocks corresponding to the Jaccard similarity between the attributes of different types (where each attribute is identified with a set of nodes)."""
    num_types = len(included_attr_types)
    block_dict = dict()
    for i in range(num_types):
        for j in range(i, num_types):
            print_flush("\nMaking (%s, %s) block..." % (included_attr_types[i], included_attr_types[j]))
            block_dict[(i, j)] = timeit(make_block)(attr_analyzer, attrs_to_nodes, included_attr_types[i], included_attr_types[j], sim = sim, delta = delta)
    block_grid = []
    for i in range(num_types):
        row = []
        for j in range(num_types):
            if (j < i):
                row.append(block_dict[(j, i)].transpose())
            else:
                row.append(block_dict[(i, j)])
        block_grid.append(row)
    return SymmetricSparseLinearOperator(BlockSparseLinearOperator(block_grid))


def main():
    p = optparse.OptionParser()
    p.add_option('--attr_type', '-a', type = str, help = 'attribute type to exclude')
    p.add_option('-p', type = str, default = 'NPMI1s', help = 'PMI type (PMIs, NPMI1s, or NPMI2s)')
    p.add_option('-e', type = str, default = 'adj', help = 'embedding (adj, normlap, regnormlap)')
    #p.add_option('-s', action = 'store_true', default = False, help = 'normalize in sphere')
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
    #report_folder = 'gplus0_lcc/reports/PMI/joint/'
    plot_folder = 'gplus0_lcc/plots/PMI/joint/'
    file_prefix1 = ('%s_%s_%s_delta' % (attr_type, sim, embedding)) + str(delta) + ('_k%d' % k)

    print_flush("\nLoading AttributeAnalyzer...")
    a = AttributeAnalyzer()
    for at in other_attr_types:
        a.load_pairwise_freq_analyzer(at)
    a.make_attrs_by_node_by_type()
    n = sum([a.pairwise_freq_analyzers[at].num_vocab for at in other_attr_types])
    tol = (1.0 / n) if (tol is None) else tol  # use 1/n instead of machine precision as default tolerance

    print_flush("\nLoading AttrsToNodes...")
    atn = AttrsToNodes()
    atn.make_nodes_by_attr_by_type(load = True, save = False)

    try:
        print_flush("\nLoading eigenvalues from '%s%s_eigvals.csv'..." % (data_folder, file_prefix1))
        eigvals = np.loadtxt('%s%s_eigvals.csv' % (data_folder, file_prefix1), delimiter = ',')
        print_flush("\nLoading embedded features from '%s%s_features_by_attr_type.pickle'..." % (data_folder, file_prefix1))
        features_by_attr_type = pickle.load(open('%s%s_features_by_attr_type.pickle' % (data_folder, file_prefix1), 'rb'))
    except FileNotFoundError:
        print_flush("Failed to load.")
        print_flush("\nComputing joint similarity operator (%s)..." % sim)
        joint_op = make_joint_embedding_operator(a, atn, other_attr_types, sim = sim, delta = delta)
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
        start_indices = [0] + list(np.cumsum([a.pairwise_freq_analyzers[at].num_vocab for at in other_attr_types]))
        features_by_attr_type = dict()
        for (i, at) in enumerate(other_attr_types):
            attr_indices, attr_vocab = get_attr_indices(a.pairwise_freq_analyzers[at], a.attributed_nodes)
            features_by_attr_type[at] = features[start_indices[i] : start_indices[i + 1], :][attr_indices, :] 
        np.savetxt('%s%s_eigvals.csv' % (data_folder, file_prefix1), eigvals, delimiter = ',')
        pickle.dump(features_by_attr_type, open('%s%s_features_by_attr_type.pickle' % (data_folder, file_prefix1), 'wb'))
    # if sphere:  # normalize the features to have unit norm (better for kMeans)
    #     for at in other_attr_types:
    #         features_for_type = features_by_attr_type[at]
    #         for i in range(features_for_type.shape[0]):            
    #             features_for_type[i] = normalize(features_for_type[i])

    if save_plot:
        print_flush("\nSaving scree plot to '%s%s_screeplot.png'..." % (plot_folder, file_prefix1))
        scree_plot(eigvals, show = False, filename = '%s%s_screeplot.png' % (plot_folder, file_prefix1))

    print_flush("\nDone!")


if __name__ == "__main__":
    main()