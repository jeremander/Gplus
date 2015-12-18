"""Computes eigenvalues and eigenvectors of the NPMI1 sparse matrices for a given attribute type."""

import pickle
import time
import pandas as pd
import optparse
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

def get_attr_indices(pfa):
    """Given a PairwiseFreqAnalyzer, returns list of vocab indices that are not unknown, as well as the vocab items themselves."""
    attr_indices, attr_vocab = [], []
    for (i, v) in enumerate(pfa.vocab):
        if (not v.startswith('*???*')):
            attr_indices.append(i)
            attr_vocab.append(v)
    return (attr_indices, attr_vocab)

def generate_cluster_report(attr_analyzer, attr_type, cluster_labels, topN = 30):
    """Given the AttributeAnalyzer, attr_type, and a list of cluster labels (corresponding to the attribute vocab indices only), generates a report listing the top N members of each cluster, and the frequency and prevalence (relative frequency) of each attribute in the data set. Orders the clusters by total occurrences of attributes in each cluster. If topN = None, list all the attributes in each cluster."""
    attr_freq_dict = attr_analyzer.attr_freqs_by_type[attr_type]
    total_attr_freqs = sum(attr_freq_dict.values())
    pfa = attr_analyzer.pairwise_freq_analyzers[attr_type]
    attr_indices, attr_vocab = get_attr_indices(pfa)
    unique_cluster_labels = set(cluster_labels)
    # compute vocab lists for each cluster
    attr_vocab_by_cluster = dict((lab, []) for lab in unique_cluster_labels)
    for (i, lab) in enumerate(cluster_labels):
        v = attr_vocab[i]
        freq = attr_freq_dict[v]
        attr_vocab_by_cluster[lab].append((v, freq, freq / total_attr_freqs))
    # sort vocab lists by decreasing frequencies
    for lab in unique_cluster_labels:
        attr_vocab_by_cluster[lab].sort(key = lambda item : item[1], reverse = True)
    # total number of occurrences of any attribute in each cluster
    total_freqs_by_cluster = dict((lab, sum([item[1] for item in attr_vocab_by_cluster[lab]])) for lab in unique_cluster_labels)
    info_by_cluster = dict((lab, dict()) for lab in unique_cluster_labels)
    # create a DataFrame for each cluster listing the top N vocab items in order with their frequencies and prevalences
    for lab in unique_cluster_labels:
        df = pd.DataFrame(attr_vocab_by_cluster[lab], columns = ['attribute', 'frequency', 'prevalence'])
        info_by_cluster[lab]['df'] = df if (topN is None) else df[:topN]
        info_by_cluster[lab]['size'] = len(attr_vocab_by_cluster[lab])
        info_by_cluster[lab]['totalFreq'] = total_freqs_by_cluster[lab]
        info_by_cluster[lab]['totalPrevalence'] = sum(df['prevalence'])
    # sort clusters by decreasing number of occurrences
    sorted_clusters_with_total_freqs = sorted(total_freqs_by_cluster.items(), key = lambda item : item[1], reverse = True)
    # generate report
    num_attrs = len(attr_vocab)
    s = ''
    for (lab, freq) in sorted_clusters_with_total_freqs:
        info = info_by_cluster[lab]
        width = 12 + len(str(lab))
        s += '#' * width + '\n'
        s += '# ' + 'CLUSTER ' + str(lab) + ' #\n'
        s += '#' * width + '\n\n'
        s += 'attribute prevalence  = %6d / %6d = %f\n' % (info['size'], num_attrs, info['size'] / num_attrs)
        s += 'occurrence prevalence = %6d / %6d = %f\n\n' % (info['totalFreq'], total_attr_freqs, info['totalPrevalence'])
        s += info['df'].to_string(index = False) + '\n\n\n'
    return s
    #return (info_by_cluster, sorted_clusters_with_total_freqs)


def main():
    p = optparse.OptionParser()
    p.add_option('--attr_type', '-a', type = str, help = 'attribute type')
    p.add_option('-p', type = str, help = 'PMI type (PMI or NPMI1)')
    p.add_option('-d', type = float, help = 'smoothing parameter')
    p.add_option('-k', type = int, help = 'number of eigenvalues')
    p.add_option('-c', type = int, help = 'number of kmeans clusters')
    p.add_option('-t', type = float, default = None, help = 'tolerance for eigsh')
    opts, args = p.parse_args()

    attr_type = opts.a
    sim = opts.p
    delta = opts.d
    k = opts.k
    nclusts = opts.c
    tol = opts.t

    print("Loading AttributeAnalyzer...")
    a = AttributeAnalyzer()
    a.load_pairwise_freq_analyzer(attr_type)
    a.load_attrs_by_node_by_type()


    print("Loading data...")
    df = pd.read_csv("gplus0_lcc/data/npmi1s_%s.dat" % opts.attr_type, header = None, sep = ' ')
    df.columns = ['row', 'col', 'data']
    n = max(max(df['row']) + 1, max(df['col']) + 1)
    print("Constructing matrix...")
    mat = coo_matrix((df['data'], (df['row'], df['col'])), shape = (n, n)).tocsr()
    mat = mat + mat.transpose().tocsr() - diags(mat.diagonal(), offsets = 0).tocsr()  # symmetrize
    start_time = time.time()
    print("Computing eigenvectors...")
    (eigenvals, eigenvecs) = eigsh(mat, k = opts.k, tol = opts.t)
    elapsed_time = time.time() - start_time
    print("Time = %.3f seconds" % elapsed_time)
    pickle.dump((eigenvals, eigenvecs), open("gplus0_lcc/data/npmi1s_%s_eigendecomp%d.pickle" % (opts.a, opts.k), 'wb'))


if __name__ == "__main__":
    main()