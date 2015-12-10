import pandas as pd
import numpy as np
import igraph as ig
import louvain
from importlib import reload
from collections import defaultdict
from scipy.sparse import coo_matrix
from autoreadwrite import *
from ggplot import *

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

class Gplus(ig.Graph, ObjectWithReadwriteProperties):
    """Graph of Google+ data."""
    readwrite_properties = {'degree_dict' : 'pickle', 'degree_power_law' : 'pickle', 'comp_sizes' : 'csv', 'louvain_memberships' : 'csv'}
    @autoreadwrite(['_degree_dict', '_degree_power_law'], ['pickle', 'pickle'])
    def degrees(self, save = False):
        """Creates dictionary of node degrees."""
        self._degree_dict = dict((v.index, v.degree()) for v in self.vs)
        self._degree_power_law = ig.power_law_fit(list(self.degree_dict.values()))
    @autoreadwrite(['_comp_sizes'], ['csv'])
    def component_sizes(self, save = False):
        """Computes connected component sizes."""
        components = self.components()
        self._comp_sizes = pd.DataFrame([len(components.subgraph(i)) for i in range(len(components))], columns = ['componentSize'])
    # @autoreadwrite(['fastgreedy_comms'], ['pickle'])
    # def fastgreedy_communities(self, save = False):
    #     """Computes communities obtained by iGraph's fastgreedy algorithm (approximately linear time)."""
    #     self._fastgreedy_comms = self.community_fastgreedy()
    @autoreadwrite(['_louvain_memberships'], ['csv'])
    def louvain(self, save = False):
        """Computes cluster memberships returned by the Louvain method (implemented in C++ via louvain-igraph package)."""
        self._louvain_memberships = pd.DataFrame(louvain.find_partition(self, method = 'Modularity').membership, columns = ['louvainMembership'])
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


class AttributeAnalyzer(ObjectWithReadwriteProperties):
    """Class for analyzing node attributes from each of the four types (school, major, employer, places_lived)."""
    readwrite_properties = {'attr_pair_freqs' : 'pickle'}
    @timeit
    def __init__(self, folder = 'gplus0_lcc/data'):
        super().__init__(folder)
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
        node_attr_filename = folder + '/node_attributes.csv'
        self.attr_df = pd.read_csv(node_attr_filename, sep = ';')
        self.attr_df['attributeVal'] = self.attr_df['attributeVal'].astype(str)
        self.attr_types = ['school', 'major', 'employer', 'places_lived']
        self.attr_dicts = dict((attr_type, read_dict(folder + '/' + attr_type + '_map.dat')) for attr_type in self.attr_types)
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
    @autoreadwrite(['attr_pair_freqs'], ['pickle'])
    def make_attr_pair_freqs(self, save = False):
        """Returns dictionary of edge counts for each pair of (non-annotated) attributes, for each attribute type."""
        if (not hasattr(self, 'attrs_by_node_by_type')):
            self.attrs_by_node_by_type = dict((attr_type, defaultdict(set)) for attr_type in self.attr_types)
            for (i, node, attr_type, attr_val) in self.attr_df.itertuples():
                self.attrs_by_node_by_type[attr_type][node].add(attr_val)
        with open(self.folder + '/undirected_edges.dat', 'r') as f:
            self.nodes = set()
            self._attr_pair_freqs = dict((attr_type, defaultdict(int)) for attr_type in self.attr_types)
            for (i, line) in enumerate(f):
                v1, v2 = [int(token) for token in line.split()[:2]]
                self.nodes.add(v1)
                self.nodes.add(v2)
                for attr_type in self.attr_types:
                    attrs_by_node = self.attrs_by_node_by_type[attr_type]
                    pair_freqs = self._attr_pair_freqs[attr_type]
                    if (v1 in attrs_by_node):
                        if (v2 in attrs_by_node):
                            for val1 in attrs_by_node[v1]:
                                for val2 in attrs_by_node[v2]:
                                    pair_freqs[tuple(sorted([val1, val2]))] += 1
                        else:
                            for val1 in attrs_by_node[v1]:
                                pair_freqs[tuple(sorted([val1, '*???*']))] += 1  # unknown attribute marker
                    else:
                        if (v2 in attrs_by_node):
                            for val2 in attrs_by_node[v2]:
                                pair_freqs[tuple(sorted(['*???*', val2]))] += 1
                        else:
                            pair_freqs[('*???*', '*???*')] += 1
            # for attr_type in self.attr_types:
            #     nodes_without_attrs = self.nodes.difference(set(self.attr_df[self.attr_df['attributeType'] == attr_type]['node']))
            #     self.attr_freqs_by_type[attr_type]['*???*'] = len(nodes_without_attrs)
    @classmethod
    def from_data(cls, folder = 'gplus0_lcc/data'):
        """Loads in files listing the node attributes for each type. The first 500 are hand-annotated. Represents each attribute type as a dictionary mapping original attributes to annotated attributes (or None if not annotated)."""
        return cls(folder)


class PairwiseFreqAnalyzer(object):
    """Manages statistics related to unordered pairwise frequencies, such as pointwise mutual information."""
    def __init__(self, vocab, pairwise_freqs, ignore_unknown = False):
        """Takes list of vocabulary and a dictionary mapping vocab pairs to counts. The order of the pairs is ignored, and any pairs with an element not in the vocab list are also ignored. If ignore_unknown = True, then the special token *???* will be dropped from the vocab list; otherwise, it will be included."""
        self.ignore_unknown = ignore_unknown
        self.vocab_set = set(vocab)  # for fast set membership querying
        if (not self.ignore_unknown):
            self.vocab_set.add('*???*')
        self.vocab = sorted([v for v in self.vocab_set]) # the canonical sorted vocab list (corresponds to indices of matrix)
        self.vocab_indices = dict((v, i) for (i, v) in enumerate(self.vocab))  # maps vocab items to canonical indices
        self.num_vocab = len(self.vocab)
        self.num_possible_pairs = (self.num_vocab * (self.num_vocab + 1)) // 2
        self.vocab_freqs = dict((v, 0) for v in self.vocab)
        self.pairwise_freqs = dict()
        for (pair, freq) in pairwise_freqs.items():
            if ((pair[0] in self.vocab_set) and (pair[1] in self.vocab_set) and (freq > 0)):
                sorted_pair = tuple(sorted(pair))
                self.vocab_freqs[sorted_pair[0]] += freq
                if (sorted_pair[1] != sorted_pair[0]):
                    self.vocab_freqs[sorted_pair[1]] += freq
                if (sorted_pair in self.pairwise_freqs):
                    self.pairwise_freqs[sorted_pair] += freq
                else:
                    self.pairwise_freqs[sorted_pair] = freq
        self.total_edges = sum(self.pairwise_freqs.values())
    def empirical_freq(self, *items, delta = 0):
        """Returns edge count of item. If two arguments are given, returns the observed count of the pair, disregarding order. If one argument is given, returns the observed count of the singleton. If delta > 0, adds delta to the counts."""
        assert(len(items) in [1, 2])
        if any([x not in self.vocab_set for x in items]):
            raise ValueError("Entries must be in the vocabulary.")
        if (len(items) == 1):
            freq = self.vocab_freqs[items[0]] + delta * self.num_vocab
            return freq
        key = tuple(sorted(items))
        try:
            freq = self.pairwise_freqs[key]
        except KeyError:
            freq = 0
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
    def to_sparse_matrix(self, sim = 'PMIs'):
        """Returns a sparse similarity/dissimilarity matrix of the vocabulary items that co-occur. Options are 'PMIs', 'PMId', 'NPMI1s', 'NPMI1d', 'NPMI2s', 'NPMI2d', which have different ranges. No smoothing is done as of now, since that would ruin the sparsity."""
        n = len(self.pairwise_freqs)
        sim_func = self.__class__.__dict__[sim]
        rows, cols, data = np.zeros(n, dtype = int), np.zeros(n, dtype = int), np.zeros(n, dtype = float)
        for (i, (v1, v2)) in enumerate(self.pairwise_freqs):
            rows[i], cols[i] = self.vocab_indices[v1], self.vocab_indices[v2]
            data[i] = sim_func(self, v1, v2)
        return coo_matrix((data, (rows, cols)), shape = (self.num_vocab, self.num_vocab))
    def to_weighted_graph(self, sim = 'NPMI1s'):
        """Returns a weighted graph of the vocabulary items, where edge weights are similarities. The sparsity of the similarity score matrix implies this graph will be sparse."""
        assert (sim in ['NPMI1s', 'NPMI2s'])
        g = ig.Graph()
        mat = self.to_sparse_matrix(sim)
        g.add_vertices(self.num_vocab)



