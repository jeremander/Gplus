import pandas as pd
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import louvain
import tempfile
import string
from importlib import reload
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import RandomizedPCA
from scipy.sparse.linalg import eigsh
from scipy.sparse import *
from autoreadwrite import *
from ggplot import *
from linop import *

def unescape(val, tabs = False):
    """Represent a string in unescaped form. If tabs = False, replaces tabs with spaces."""
    s = str(val).encode().decode('unicode-escape')
    if (not tabs):
        s = s.replace('\t', ' ')
    return s

def safe_divide(num, den):
    """Floating point division, with the convention that 0 / 0 = 0, (+/-)c / 0 = (+/-)inf for c > 0."""
    if (num == 0.0):
        return 0.0
    if (den == 0.0):
        return np.sign(num) * float('inf')
    return (num / den)

def normalize(vec):
    """Normalizes a vector to have unit norm."""
    return vec / np.linalg.norm(vec)

def scree_plot(eigvals, show = True, filename = None):
    """Makes a scree plot of the absolute value of a series of eigenvalues using matplotlib. If show = True, displays the plot. If filename is not None, saves the plot to this filename."""
    abs_eigvals = np.abs(eigvals)
    ranked_abs_eigvals = np.array(sorted(abs_eigvals, reverse = True))
    plt.plot(ranked_abs_eigvals, linewidth = 3)
    ax = plt.axes()
    ax.set_title('Scree plot of eigenvalues')
    ax.set_xlabel('rank', labelpad = 10)
    ax.set_ylabel('abs(eigenvalue)', labelpad = 15)
    ax.set_ylim(ymin = 0)
    if filename:
        plt.savefig(filename)
    if show:
        plt.show(block = False)


class Gplus(ig.Graph, ObjectWithReadwriteProperties):
    """Graph of Google+ data."""
    num_vertices = {'gplus0_lcc' : 4690159}  # number of vertices in data set
    readwrite_properties = {'degree_dict' : 'pickle', 'degree_power_law' : 'pickle', 'comp_sizes' : 'csv', 'louvain_memberships' : 'csv', 'sparse_adjacency_operator' : 'pickle'}
    def __init__(self, folder = 'gplus0_lcc/data'):
        ObjectWithReadwriteProperties.__init__(self, folder)
    @autoreadwrite(['degree_dict', 'degree_power_law'], ['pickle', 'pickle'])
    def degrees(self, load = True, save = False):
        """Creates dictionary of node degrees."""
        self._degree_dict = dict((v.index, v.degree()) for v in self.vs)
        self._degree_power_law = ig.power_law_fit(list(self.degree_dict.values()))
    @autoreadwrite(['comp_sizes'], ['csv'])
    def component_sizes(self, load = True, save = False):
        """Computes connected component sizes."""
        components = self.components()
        self._comp_sizes = pd.DataFrame([len(components.subgraph(i)) for i in range(len(components))], columns = ['componentSize'])
    # @autoreadwrite(['fastgreedy_comms'], ['pickle'])
    # def fastgreedy_communities(self, save = False):
    #     """Computes communities obtained by iGraph's fastgreedy algorithm (approximately linear time)."""
    #     self._fastgreedy_comms = self.community_fastgreedy()
    @autoreadwrite(['louvain_memberships'], ['csv'])
    def louvain(self, load = True, save = False):
        """Computes cluster memberships returned by the Louvain method (implemented in C++ via louvain-igraph package)."""
        self._louvain_memberships = pd.DataFrame(louvain.find_partition(self, method = 'Modularity').membership, columns = ['louvainMembership'])
    @autoreadwrite(['sparse_adjacency_operator'], ['pickle'])
    def make_sparse_adjacency_operator(self, load = True, save = False):
        """Reads the graph from edge list, and returns it as a SymmetricSparseLinearOperator object."""
        filename = self.folder + '/undirected_edges.dat'
        print("\nLoading data from '%s'..." % filename)
        edges = pd.read_table(filename, header = None, sep = ' ')
        n = max(max(edges[0]) + 1, max(edges[1]) + 1)
        A = coo_matrix((np.ones(len(edges)), (edges[0], edges[1])), shape = (n, n)).tocsr()
        A = A + A.transpose().tocsr() - diags(A.diagonal(), offsets = 0).tocsr()  # symmetrize the matrix
        self._sparse_adjacency_operator = SymmetricSparseLinearOperator(A)
    def make_graph_embedding_matrix(self, embedding = 'adj', k = 50, tol = None, plot = False, load = True, save = False):
        """Makes graph embedding matrix, keeping features of only the nodes possessing at least one attribute. embedding options are adjacency matrix (adj), diagonal-added adjacency matrix (adj+diag), normalized Laplacian (normlap), and regularized normalized Laplacian (regnormlap)."""
        assert (embedding in ['adj', 'adj+diag', 'normlap', 'regnormlap'])
        obj_name = '%s_k%d_graph_embedding_matrix' % (embedding, k)
        did_load = False
        if load:
            try:
                eigvals = np.loadtxt('%s/%s_k%d_graph_eigvals.csv' % (folder, embedding, k), delimiter = ',')
                self.graph_embedding_matrix = timeit(load_object)(self.folder, obj_name, 'pickle')
                did_load = True
            except:
                print("\nCould not load %s from file." % obj_name)
        if (not did_load):
            node_attr_filename = self.folder + '/node_attributes.csv'
            attr_df = pd.read_csv(node_attr_filename, sep = ';')
            self.attributed_nodes = sorted(list(set(attr_df['node'])))
            A = self.sparse_adjacency_operator
            n = A.shape[0]
            tol = 1.0 / n if (tol is None) else tol
            matrix_type = 'adjacency' if (embedding == 'adj') else ('diagional-added adjacency' if (embedding == 'diag+adj') else ('normalized Laplacian' if (embedding == 'normlap') else 'regularized normalized Laplacian'))
            print("\nComputing k = %d eigenvectors of %d x %d %s matrix..." % (n, n, matrix_type, k))
            if (embedding == 'adj'):
                (eigvals, features) = timeit(eigsh)(A, k = k, tol = tol)
                features = np.sqrt(np.abs(eigvals)) * features  # scale the feature columns by the sqrt of the eigenvalues
            elif (embedding == 'adj+diag'):
                adj_diag = SparseDiagonalAddedAdjacencyOperator(A)
                (eigvals, features) = timeit(eigsh)(adj_diag, k = k, tol = tol)
            elif (embedding == 'normlap'):
                normlap = SparseNormalizedLaplacian(A)
                (eigvals, features) = timeit(eigsh)(normlap, k = k, tol = tol)
            elif (embedding == 'regnormlap'):
                regnormlap = SparseRegularizedNormalizedLaplacian(A)
                (eigvals, features) = timeit(eigsh)(regnormlap, k = k, tol = tol)
            self.graph_embedding_matrix = features[self.attributed_nodes, :]
        if save:
            if plot:
                scree_plot(eigvals, show = False, filename = '%s/%s_k%d_graph_screeplot' % (self.folder.replace('data', 'plots'), embedding, k))
            np.savetxt('%s/%s_k%d_graph_eigvals.csv' % (self.folder, embedding, k), eigvals, delimiter = ',')
            timeit(save_object)(self.graph_embedding_matrix, self.folder, obj_name, 'pickle')
    def __len__(self):
        return len(self.vs)
    @classmethod 
    def from_data(cls, folder = 'gplus0_lcc/data', directed = False):
        """Reads the graph from edge list."""
        filename = folder + '/%sdirected_edges.dat' % ('' if directed else 'un')
        print("Loading data from '%s'..." % filename)
        g = timeit(cls.Read_Edgelist)(filename, directed = directed)
        print("Successfully loaded data.")
        if directed:
            print("Removing directions from edges...")
            timeit(Gplus.to_undirected)(g)
        g.folder = folder
        return g


class PairwiseFreqAnalyzer(object):
    """Manages statistics related to unordered pairwise frequencies, such as pointwise mutual information. Represents the pairwise frequencies as a sparse matrix of counts of each pair using the DOK (dictionary of keys) format, then converts it to CSR (compressed sparse row) format."""
    def __init__(self, vocab):
        """Constructs PairwiseFreqAnalyzer object with a list of vocab. Creates a mapping from vocab to indices for the sparse matrix representation.""" 
        self.vocab = vocab  # the canonical vocab list (in order of matrix indices)
        self.vocab_indices = dict((v, i) for (i, v) in enumerate(self.vocab))  # maps vocab items to canonical indices
        self.vocab_freqs = dict((v, 0) for v in self.vocab)  # counts number of edges seen with each vocab word in it
        #self.vocab_set = set(vocab)  # for fast set membership querying
        self.num_vocab = len(self.vocab)
        self.num_possible_pairs = (self.num_vocab * (self.num_vocab + 1)) // 2
        self.freq_mat = dok_matrix((self.num_vocab, self.num_vocab), dtype = np.int64)
    def add_pair(self, pair):
        """Given a pair of objects, adds one to the count of the object pair. If an item in the pair is not in the vocab list, raises a KeyError."""
        v1, v2 = pair
        [i, j] = sorted([self.vocab_indices[v1], self.vocab_indices[v2]])
        self.freq_mat[i, j] += 1
    def finalize_construction(self):
        """Once all pairs are added, performs some computations and converts the sparse matrix from dok to csr format."""
        self.total_edges = sum(self.freq_mat.values())
        self.freq_mat = self.freq_mat.tocsr()
        sym_freq_mat = self.freq_mat + self.freq_mat.transpose().tocsr() - diags(self.freq_mat.diagonal(), offsets = 0).tocsr()  # symmetrize the matrix
        for (i, v) in enumerate(self.vocab):
            self.vocab_freqs[v] = sym_freq_mat[i,:].data.sum()
    def empirical_freq(self, *items, delta = 0):
        """Returns edge count of item. If two arguments are given, returns the observed count of the pair, disregarding order. If one argument is given, returns the observed count of the singleton. If delta > 0, adds delta to the counts."""
        assert(len(items) in [1, 2])
        if any([x not in self.vocab_freqs for x in items]):
            raise ValueError("Entries must be in the vocabulary.")
        if (len(items) == 1):
            freq = self.vocab_freqs[items[0]] + delta * self.num_vocab
            return freq
        i, j = sorted([self.vocab_indices[items[0]], self.vocab_indices[items[1]]])
        freq = self.freq_mat[i, j]
        return (freq + delta)
    def empirical_prob(self, *items, delta = 0):
        """Returns empirical probability of item. If two arguments are given, this is the smoothed number of occurrences of the pair divided by the smoothed number of edges, under add-delta smoothing. If one argument is given, this is the smoothed number of occurrences of the item in any pair divided by the smoothed number of edges."""
        denominator = self.total_edges + delta * self.num_possible_pairs
        return self.empirical_freq(*items, delta = delta) / denominator
    def conditional_prob(self, item1, item2, delta = 0):
        return safe_divide(self.empirical_freq(item1, item2, delta = delta), self.empirical_freq(item2, delta = delta))
    def PMIs(self, item1, item2, delta = 0):
        """Pointwise mutual information of two items. Ranges from -inf to -log p(x,y) at most, with 0 for independence"""
        return np.log(self.empirical_prob(item1, item2, delta = delta)) - np.log(self.empirical_prob(item1, delta = delta)) - np.log(self.empirical_prob(item2, delta = delta))
    def PMId(self, item1, item2, delta = 0):
        """Negative pointwise mutual information of two items. Ranges from log p(x,y) to inf, with 0 for independence."""
        return -self.PMIs(item1, item2, delta = delta)
    def NPMI1s(self, item1, item2, delta = 0):
        """PMI normalized so that it is a similarity score ranging from 0 to 1, with 1/2 for independence."""
        return safe_divide(np.log(self.empirical_prob(item1, delta = delta)) + np.log(self.empirical_prob(item2, delta = delta)), 2.0 * np.log(self.empirical_prob(item1, item2, delta = delta)))
    def NPMI1d(self, item1, item2, delta = 0):
        """PMI normalized so that it is a dissimilarity score ranging from 0 to 1, with 1/2 for independence."""
        return 1.0 - self.NPMI1s(item1, item2, delta = delta)
    def NPMI2s(self, item1, item2, delta = 0):
        """PMI transformed so that it is a similarity score ranging from 0 to inf, with 1 for independence."""
        return -np.log(1.0 - self.NPMI1s(item1, item2, delta = delta)) / np.log(2.0)
    def NPMI2d(self, item1, item2, delta = 0):
        """PMI transformed so that it is a dissimilarity score ranging from 0 to inf, with 1 for independence."""
        return -np.log(self.NPMI1s(item1, item2, delta = delta)) / np.log(2.0)
    @timeit
    def to_sparse_PMI_matrix(self, sim = 'PMIs', symmetric = True):
        """Returns a sparse similarity/dissimilarity matrix of the PMIs of vocabulary items that co-occur. Options are 'PMIs', 'PMId', 'NPMI1s', 'NPMI1d', 'NPMI2s', 'NPMI2d', which have different ranges. No smoothing is done yet, since that would ruin the sparsity."""
        n = len(self.freq_mat.data)
        sim_func = self.__class__.__dict__[sim]
        log_single_freqs = np.log(np.array([self.empirical_freq(self.vocab[i]) for i in range(self.num_vocab)]))
        log_total_edges = np.log(self.total_edges)
        data = np.zeros(n, dtype = float)
        coo = self.freq_mat.tocoo()  # convert to coo_matrix
        coo.data = np.log(coo.data)  # store the log-frequencies
        # efficiently compute the score
        for (k, (i, j, log_freq)) in enumerate(zip(coo.row, coo.col, coo.data)):
            if (sim == 'PMIs'):
                data[k] = log_freq - log_single_freqs[i] - log_single_freqs[j] + log_total_edges
            elif (sim == 'NPMI1s'):
                data[k] = (log_single_freqs[i] + log_single_freqs[j] - 2 * log_total_edges) / (2 * (log_freq - log_total_edges))
            else:
                data[k] = sim_func(self, self.vocab[i], self.vocab[j])
        mat = coo_matrix((data, (coo.row, coo.col)), shape = (self.num_vocab, self.num_vocab)).tocsr()
        if symmetric:  # symmetrize the matrix
            mat = mat + mat.transpose().tocsr() - diags(mat.diagonal(), offsets = 0).tocsr()
        return mat
    @timeit
    def to_sparse_PMI_operator(self, sim = 'NPMI1s', delta = 0):
        """Returns a LinearOperator object encoding the sparse + low-rank representation of the PMI similarity matrix. This can be used in place of an actual matrix in various computations. If sim != 'PMIs', can use an alternative formulation of PMI, but only if delta = 0."""
        assert (sim in ['PMIs', 'NPMI1s', 'NPMI2s'])
        assert ((delta > 0) if (sim == 'PMIs') else (delta == 0))
        if (sim != 'PMIs'):  # just use the sparse NPMI matrix with no smoothing
            csr_mat = self.to_sparse_PMI_matrix(sim)
            return SymmetricSparseLinearOperator(csr_mat)
        log_delta = np.log(delta)
        coo = self.freq_mat.tocoo()
        data = np.log(coo.data + delta) - log_delta
        F = coo_matrix((data, (coo.row, coo.col)), shape = (self.num_vocab, self.num_vocab)).tocsr()
        F = F + F.transpose().tocsr() - diags(F.diagonal(), offsets = 0).tocsr()  # symmetrize the matrix
        u = np.log(np.array([self.empirical_freq(self.vocab[i], delta = delta) for i in range(self.num_vocab)]))
        Delta = log_delta + np.log(self.total_edges + delta * self.num_possible_pairs)
        return PMILinearOperator(F, Delta, u)
    @timeit
    def to_PMI_weighted_graph(self, sim = 'NPMI1s'):
        """Returns a weighted graph of the vocabulary items, where edge weights are similarities. The sparsity of the similarity score matrix implies this graph will be sparse."""
        assert (sim in ['NPMI1s', 'NPMI2s'])
        mat = self.to_sparse_PMI_matrix(sim, symmetric = False).tocoo()  # graph will be undirected, so don't need symmetry
        with tempfile.TemporaryFile(mode = 'w+') as f:
            for (i, j) in zip(mat.row, mat.col):
                f.write("%d %d\n" % (i, j))
            f.seek(0)
            g = ig.Graph.Read_Edgelist(f, directed = False)
        g.es['weight'] = mat.data
        return g

def get_attr_indices(pfa, attributed_nodes = None):
    """Given a PairwiseFreqAnalyzer, returns list of vocab indices that are not unknown, as well as the vocab items themselves. If attributed_nodes is not None, retains the unknown attributes corresponding to the given nodes that have attributes (in other attribute types)."""
    attributed_node_set = set() if (attributed_nodes is None) else set(attributed_nodes)
    attributed_node_vocab_set = set(('*???*_%d' % n) for n in attributed_node_set)
    attr_indices, attr_vocab = [], []
    for (i, v) in enumerate(pfa.vocab):
        if ((v in attributed_node_vocab_set) or (not v.startswith('*???*'))):
            attr_indices.append(i)
            attr_vocab.append(v)
    return (attr_indices, attr_vocab)


class AttributeAnalyzer(ObjectWithReadwriteProperties):
    """Class for analyzing node attributes from each of the four types (school, major, employer, places_lived)."""
    readwrite_properties = {'pairwise_freq_analyzers' : 'pickle', 'attrs_by_node_by_type' : 'pickle'}
    @timeit
    def __init__(self, dataset = 'gplus0_lcc', load_data = True):
        folder = dataset + '/data'
        super().__init__(folder)
        self.num_vertices = Gplus.num_vertices[dataset]  # need to know how many vertices are in the dataset
        if load_data:
            self.load_data()
    def load_data(self):
        def read_dict(filename):
            """Reads a string dictionary from a file with the following format: on each line, the key comes first, then a tab followed by a values. The keys & values may be delimited by double quotes in case spaces are present. Only lines with both key and value will be present."""
            d = dict()
            with open(filename, 'r', encoding = 'utf-8') as f:
                for line in f:
                    tokens = [token.strip('"') for token in line.strip().split('\t')]
                    if (len(tokens) > 1):
                        key, val = tokens[0], tokens[1]
                        d[key] = val
            return d
        node_attr_filename = self.folder + '/node_attributes.csv'
        self.attr_df = pd.read_csv(node_attr_filename, sep = ';')
        self.attr_df['attributeVal'] = self.attr_df['attributeVal'].astype(str)
        self.attributed_nodes = sorted(list(set(self.attr_df['node'])))
        self.attr_types = ['employer', 'major', 'places_lived', 'school']
        self.attr_dicts = dict((attr_type, read_dict(self.folder + '/' + attr_type + '_map.dat')) for attr_type in self.attr_types)
        self.attr_map = dict((attr_type, lambda attr, attr_type = attr_type : self.attr_dicts[attr_type][attr] if (attr in self.attr_dicts[attr_type]) else attr) for attr_type in self.attr_types)
        self.attr_freqs_by_type = dict((t, defaultdict(int)) for t in self.attr_types)
        self.annotated_attr_freqs_by_type = dict((t, defaultdict(int)) for t in self.attr_types)
        for (t, val) in zip(self.attr_df['attributeType'], self.attr_df['attributeVal']):
            self.attr_freqs_by_type[t][val] += 1
            self.annotated_attr_freqs_by_type[t][self.attr_map[t](val)] += 1
        self.num_unique_attrs_by_type = dict((t, len(self.attr_freqs_by_type[t])) for t in self.attr_types)
        self.num_attr_instances_by_type = dict((t, sum(self.attr_freqs_by_type[t].values())) for t in self.attr_types)
        self.sorted_attr_freqs_by_type = dict((t, sorted(self.attr_freqs_by_type[t].items(), key = lambda pair : pair[1], reverse = True)) for t in self.attr_types)
        self.sorted_annotated_attr_freqs_by_type = dict((t, sorted([item for item in self.annotated_attr_freqs_by_type[t].items() if (item[0] in self.annotated_attr_freqs_by_type[t])], key = lambda pair : pair[1], reverse = True)) for t in self.attr_types)
        for t in self.attr_types:
            self.sorted_annotated_attr_freqs_by_type[t] += [item for item in self.sorted_attr_freqs_by_type[t] if (item[0] not in self.attr_dicts[t])]
            self.sorted_annotated_attr_freqs_by_type[t].sort(key = lambda pair : pair[1], reverse = True)
    def attr_freq_df(self, rank_thresh = 100):
        afdf = pd.DataFrame(columns = ['rank', 'freq', 'percentage', 'type', 'annotated'])
        for annotated in [False, True]:
            for t in self.attr_types:
                df = pd.DataFrame(columns = afdf.columns)
                df['rank'] = list(range(rank_thresh))
                saf = self.sorted_annotated_attr_freqs_by_type[t] if annotated else self.sorted_attr_freqs_by_type[t]
                df['freq'] = [pair[1] for pair in saf[:rank_thresh]]
                df['percentage'] = 100 * np.cumsum(df['freq']) / self.num_attr_instances_by_type[t]
                df['type'] = t
                df['annotated'] = annotated
                afdf = afdf.append(df)
        return afdf
    def rank_plot(self, rank_thresh = 100):
        """Returns plot of the frequencies of the attributes, sorted by rank."""
        afdf = self.attr_freq_df(rank_thresh)
        return ggplot(aes(x = 'rank', y = 'freq', color = 'type', linetype = 'annotated'), data = afdf) + geom_line(size = 3) + ggtitle("Most frequent attributes by type") + xlab("rank") + xlim(low = -1, high = rank_thresh + 1) + ylab("") + scale_y_log10() + scale_x_continuous(breaks = range(0, int(1.05 * rank_thresh), rank_thresh // 5))
    def cumulative_rank_plot(self, rank_thresh = 100):
        """Returns plot showing the cumulative proportions covered by the attributes sorted by rank."""
        afdf = self.attr_freq_df(rank_thresh)
        return ggplot(aes(x = 'rank', y = 'percentage', color = 'type', linetype = 'annotated'), data = afdf) + geom_line(size = 3) + ggtitle("Cumulative percentage of most frequent attributes") + xlim(low = -1, high = rank_thresh + 1) + ylab("%") + scale_y_continuous(labels = range(0, 120, 20), limits = (0, 100)) + scale_x_continuous(breaks = range(0, int(1.05 * rank_thresh), rank_thresh // 5))
    @timeit
    def load_pairwise_freq_analyzer(self, attr_type):
        """Loads a PairwiseFreqAnalyzer if not already owned by the object."""
        if (not hasattr(self, '_pairwise_freq_analyzers')):
            self._pairwise_freq_analyzers = dict()
        if (attr_type not in self._pairwise_freq_analyzers):
            self._pairwise_freq_analyzers[attr_type] = load_object(self.folder, 'pairwise_freq_analyzer_%s' % attr_type, 'pickle')
    @autoreadwrite(['pairwise_freq_analyzers'], ['pickle'])
    def load_all_pairwise_freq_analyzers(self):
        """Loads all PairwiseFreqAnalyzers."""
        for attr_type in self.attr_types:
            self.load_pairwise_freq_analyzer(attr_type)
    @autoreadwrite(['attrs_by_node_by_type'], ['pickle'])
    def make_attrs_by_node_by_type(self, load = True, save = False):
        self._attrs_by_node_by_type = dict((attr_type, defaultdict(set)) for attr_type in self.attr_types)
        for (i, node, attr_type, attr_val) in self.attr_df.itertuples():
                self._attrs_by_node_by_type[attr_type][node].add(attr_val)
    @autoreadwrite(['pairwise_freq_analyzers'], ['pickle'])
    def make_pairwise_freq_analyzers(self, load = True, save = False, unknown_style = 2):
        """Makes PairwiseFreqAnalyzer objects for each attribute type. These objects can be used to perform statistics on pairwise attribute counts and to compute pairwise similarity matrices between attributes. unknown_style is an integer indicating one of three ways of handling unknown attributes:
            0: Unknown attributes will not be considered (nor will attributes paired with them).
            1: The special token *???* will represent all unknown attributes.
            2: The special token *???*_i will represent the occurrence of an unknown attribute for node i.
            attr_types can optionally specify the attribute types for which to construct PairwiseFreqAnalyzers."""
        if (not hasattr(self, '_attrs_by_node_by_type')):
            self.make_attrs_by_node_by_type()
        self.unknown_style = unknown_style
        self._pairwise_freq_analyzers = dict()
        for attr_type in self.attr_types:
            attrs_by_node = self._attrs_by_node_by_type[attr_type]
            vocab = set()
            for i in range(self.num_vertices):
                if (i in attrs_by_node):
                    vocab.update(attrs_by_node[i])
                elif (self.unknown_style == 2):  # include unique unknown token for each unattributed node
                    vocab.add('*???*_%d' % i)
            if (self.unknown_style == 1):  # include a single unknown token to cover all unattributed nodes
                vocab.add('*???*')
            vocab = sorted(list(vocab)) # sort alphabetically
            self._pairwise_freq_analyzers[attr_type] = PairwiseFreqAnalyzer(vocab)
        with open(self.folder + '/undirected_edges.dat', 'r') as f:
            for (i, line) in enumerate(f):
                if (i % 100000 == 0):
                    print(i)
                v1, v2 = [int(token) for token in line.split()[:2]]
                for attr_type in self.attr_types:
                    attrs_by_node = self.attrs_by_node_by_type[attr_type]
                    if (v1 in attrs_by_node):
                        if (v2 in attrs_by_node):
                            for val1 in attrs_by_node[v1]:
                                for val2 in attrs_by_node[v2]:
                                    self._pairwise_freq_analyzers[attr_type].add_pair((val1, val2))
                        else:
                            for val1 in attrs_by_node[v1]:
                                self._pairwise_freq_analyzers[attr_type].add_pair((val1, ('*???*_%d' % v2) if (self.unknown_style == 2) else '*???*'))
                    else:
                        if (v2 in attrs_by_node):
                            for val2 in attrs_by_node[v2]:
                                self._pairwise_freq_analyzers[attr_type].add_pair((('*???*_%d' % v1) if (self.unknown_style == 2) else '*???*', val2))
                        else:
                            self._pairwise_freq_analyzers[attr_type].add_pair((('*???*_%d' % v1) if (self.unknown_style == 2) else '*???*', ('*???*_%d' % v2) if (self.unknown_style == 2) else '*???*'))
        for attr_type in self.attr_types:
           self._pairwise_freq_analyzers[attr_type].finalize_construction()
    @timeit
    def make_count_vectorizers(self, max_count_features = 500, load = True, save = False):
        """Makes CountVectorizer objects for each of the attribute types. These can be used to give sparse vector representations of each attribute. Makes two separate CountVectorizers for each type, one for words and one for characters. max_features is the maximum number of features to include for each CountVectorizer."""
        obj_name = "count_vectorizers%d" % max_count_features
        did_load = False
        if load:
            try:
                (self.word_cvs, self.char_cvs) = timeit(load_object)(self.folder, obj_name, 'pickle')
                did_load = True
            except:
                print("Could not load %s from file.\n" % obj_name)
        if (not did_load):
            print("Constructing from scratch...")
            def make_vectorizers():
                self.max_count_features = max_count_features
                self.word_cvs, self.char_cvs = dict(), dict()
                self.word_features, self.char_features = dict(), dict()
                self.indices_by_attr_by_type = dict()  # save off canonical mapping from attributes to indices
                for attr_type in self.attr_types:
                    # word vectorizer: remove accents & punctuation, split by whitespace, include single-character words
                    self.word_cvs[attr_type] = CountVectorizer('content', strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\b\w+\b', max_features = self.max_count_features)
                    attrs = [pair[0] for pair in self.sorted_attr_freqs_by_type[attr_type]]
                    self.indices_by_attr_by_type[attr_type] = dict((attr, i) for (i, attr) in enumerate(attrs))
                    self.word_features[attr_type] = csr_matrix(self.word_cvs[attr_type].fit_transform(attrs), dtype = float)
                    # char vectorizer: include all characters, including whitespace, punctuation, accents, and unicode
                    self.char_cvs[attr_type] = CountVectorizer('content', analyzer = 'char', max_features = self.max_count_features)
                    self.char_features[attr_type] = csr_matrix(self.char_cvs[attr_type].fit_transform(attrs), dtype = float)
            timeit(make_vectorizers)()
        if save:
            timeit(save_object)((self.word_cvs, self.char_cvs), self.folder, obj_name, 'pickle')
    def node_to_attr_vec(self, index, attr_type):
        """Given a node index and an attribute type, returns the vector (as csr_matrix) obtained from the word/char counts for all attributes of that type. If the node possesses no attributes, this is the zero vector. If it possesses more than one attribute, returns the mean of the attribute vectors."""
        if (index in self.attrs_by_node_by_type[attr_type]):
            attrs = self.attrs_by_node_by_type[attr_type][index]
            if (len(attrs) == 0):
                return csr_matrix((1, 2 * self.max_count_features))
            else:
                indices_by_attr = self.indices_by_attr_by_type[attr_type]
                word_features_mat, char_features_mat = self.word_features[attr_type], self.char_features[attr_type]
                word_features_vec, char_features_vec = csr_matrix((1, self.max_count_features)), csr_matrix((1, self.max_count_features))
                for attr in attrs:
                    attr_index = indices_by_attr[attr]
                    word_features_vec += word_features_mat[attr_index]
                    char_features_vec += char_features_mat[attr_index]
                vec = hstack([word_features_vec, char_features_vec])
                return (vec / len(attrs))
        else:
            return csr_matrix((1, 2 * self.max_count_features))
    def get_attribute_indicator(self, attr, attr_type):
        """Given an attribute and its type, returns a Series of indicators for the graph vertices. 0 indicates the presence of the attribute, 1 indicates the presence of other attributes but not the given attribute, and nan indicates the lack of any attributes in that type."""
        attrs_by_node = self.attrs_by_node_by_type[attr_type]
        ind = np.zeros(self.num_vertices, dtype = float)
        for i in range(self.num_vertices):
            if (i in attrs_by_node):
                attrs = attrs_by_node[i]
                if (len(attrs) == 0):
                    ind[i] = np.nan
                elif (attr in attrs):
                    ind[i] = 1.0
            else:
                ind[i] = np.nan
        return pd.Series(ind)
    def get_features_for_nodes(self, indices, attr_types):
        """Returns count feature matrix for the given graph nodes. Only the attribute types in attr_types are included."""
        return vstack([hstack([self.node_to_attr_vec(i, attr_type) for attr_type in attr_types]) for i in indices])
    def make_complete_feature_matrix(self, max_count_features = 500, load = True, save = False):
        """Makes sparse matrix of features for each node (all attribute types) based on the CountVectorizer objects."""
        obj_name = 'complete_feature_matrix%d' % max_count_features
        did_load = False
        if load:
            try:
                self.complete_feature_matrix = timeit(load_object)(self.folder, obj_name, 'pickle')
                did_load = True
            except:
                print("Could not load %s from file.\n" % obj_name)
        if (not did_load):
            print("Constructing from scratch...")
            # ensure we have count vectorizers of the right dimensions
            if ((not hasattr(self, 'word_cvs')) or (self.word_cvs['employer'].max_features != max_count_features)):
                self.make_count_vectorizers(max_count_features)
            def make_mat():
                self.complete_feature_matrix = self.get_features_for_nodes(range(self.num_vertices), self.attr_types).tocsr()
            timeit(make_mat)()
        if save:
            timeit(save_object)(self.complete_feature_matrix, self.folder, obj_name, 'pickle')
    def make_complete_embedding_matrix(self, sim = 'NPMI1s', embedding = 'eig', delta = 0.0, k = 50, load = True, save = False):
        """Makes full matrix of feature embeddings based on PMI similarities (saved off as matrix files). Rows are nodes, columns are features, grouped into blocks for each attribute type."""
        obj_name = '%s_%s_delta%s_k%d_complete_embedding_matrix' % (sim, embedding, str(delta), k)
        did_load = False
        if load:
            try:
                self.complete_embedding_matrix = timeit(load_object)(self.folder, obj_name, 'pickle')
                did_load = True
            except:
                print("\nCould not load %s from file." % obj_name)
        if (not did_load):
            shape = (self.num_vertices, len(self.attr_types) * k)
            embedding_matrix_blocks = []  # list of partial matrices by attribute type
            for attr_type in self.attr_types:
                feature_filename = self.folder + '/PMI/%s_%s_%s_delta%s_k%d_features.csv' % (attr_type, sim, embedding, str(delta), k)
                print("\nLoading features from %s..." % feature_filename)
                feature_mat = timeit(np.loadtxt)(feature_filename, delimiter = ',')
                self.load_pairwise_freq_analyzer(attr_type)
                pfa = self.pairwise_freq_analyzers[attr_type]
                (attr_indices, attr_vocab) = get_attr_indices(pfa)
                assert (len(attr_indices) == feature_mat.shape[0])  # confirm the features match
                index_by_vocab = dict((v, i) for (i, v) in enumerate(attr_vocab))  # want matrix indices for each attribute
                block = lil_matrix((self.num_vertices, k))
                attrs_by_node = self.attrs_by_node_by_type[attr_type]
                for i in range(self.num_vertices):
                    attrs = attrs_by_node[i]
                    if (len(attrs) > 0):
                        row = np.zeros(k, dtype = float)  # compute average feature vector
                        for attr in attrs:
                            row += feature_mat[index_by_vocab[attr]]
                        row /= len(attrs)
                        block[i] = row
                embedding_matrix_blocks.append(block)
                self.complete_embedding_matrix = hstack(embedding_matrix_blocks).tocsr()
        if save:
            timeit(save_object)(self.complete_embedding_matrix, self.folder, obj_name, 'pickle')
    @timeit
    def PCA_embedding_matrix(self, attr_types, n_components, normalize = True, seed = None):
        """Extracts submatrix of the embedding matrix corresponding to the desired attribute types, then uses PCA to reduce the number of columns to n_components."""
        assert hasattr(self, 'complete_embedding_matrix')
        k = self.complete_embedding_matrix.shape[1] // len(self.attr_types)
        indices = []
        for (i, attr_type) in enumerate(self.attr_types):
            if (attr_type in attr_types):
                indices += list(range(i * k, (i + 1) * k))
        mat = self.complete_embedding_matrix[:, indices]
        rows = sorted(list(set(mat.tocoo().row)))
        mat = mat[rows, :].todense()
        if normalize:  # normalize by block (only nonzero rows of each block)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1] // k):
                    segment = mat[i, j * k : (j + 1) * k]
                    norm = np.linalg.norm(segment)
                    if (np.abs(norm) > 1e-8):
                        mat[i, j * k : (j + 1) * k] = segment / norm
        pca = RandomizedPCA(n_components = n_components, random_state = seed)
        res = pca.fit_transform(mat)
        mat2 = coo_matrix((res.reshape(res.shape[0] * res.shape[1]), (np.repeat(rows, res.shape[1]), list(range(res.shape[1])) * res.shape[0])))
        return (mat, mat2.tocsr(), pca)
    def get_attribute_sample(self, attr, attr_type, n):
        """Selects a random n nodes with the given attribute, and a random n nodes without it. Returns a triple of index lists: first the n with the attribute, then the n without it, then the remaining unselected nodes whose attribute status is known."""
        ind = self.get_attribute_indicator(attr, attr_type)
        known_true = list(ind[ind==1].index)
        known_false = list(ind[ind==0].index)
        if (n > min(len(known_true), len(known_false))):
            raise ValueError("Sample size is too large.")
        training_true = sorted(list(np.random.permutation(known_true)[:n]))
        training_false = sorted(list(np.random.permutation(known_false)[:n]))
        test = sorted(list(set(known_true + known_false).difference(training_true).difference(training_false)))
        return (training_true, training_false, test)
    @timeit
    def get_training_and_test(self, attr, attr_type, n):
        """Selects an (n, n) training sample of nodes with/without the attribute, and a test set of the remainder. Returns a pair (features, outputs) for the both the training and test sets."""
        assert hasattr(self, 'complete_feature_matrix')
        (training_true, training_false, test) = self.get_attribute_sample(attr, attr_type, n)
        training = sorted(training_true + training_false)
        attr_indicator = self.get_attribute_indicator(attr, attr_type)
        # get the column indices for the desired features
        max_count_features = self.complete_feature_matrix.shape[1] // 8
        attr_type_index = self.attr_types.index(attr_type)
        attr_cols = range(2 * max_count_features * attr_type_index, 2 * max_count_features * (attr_type_index + 1))
        good_cols = sorted(list(set(range(self.complete_feature_matrix.shape[1])).difference(attr_cols)))
        return ((self.complete_feature_matrix[training][:, good_cols], attr_indicator[training]), (self.complete_feature_matrix[test][:, good_cols], attr_indicator[test]))
    @classmethod
    def from_data(cls, dataset = 'gplus0_lcc'):
        """Loads in files listing the node attributes for each type. The first 500 are hand-annotated. Represents each attribute type as a dictionary mapping original attributes to annotated attributes (or None if not annotated)."""
        return cls(dataset)


