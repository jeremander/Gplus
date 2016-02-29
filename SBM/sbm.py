import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import pymc
import fractions
import warnings
from collections import defaultdict
#from importlib import reload

class Fraction(fractions.Fraction):
    def __repr__(self):
        return str(self)

class SBM(object):
    """Class for Stochastic Block Models."""
    def __init__(self, edge_probs, block_probs = None):
        """edge_probs is symmetric matrix of communication probabilities. block_probs is vector of block probabilities."""
        assert (edge_probs.shape[0] == edge_probs.shape[1])
        assert np.allclose(edge_probs.transpose(), edge_probs)
        assert (0.0 <= edge_probs.min() <= edge_probs.max() <= 1.0)
        self.edge_probs = edge_probs
        self.m = edge_probs.shape[0]
        if (block_probs is None):
            self.block_probs = np.ones(self.m, dtype = float) / self.m  # uniform distribution
        else:
            assert (len(block_probs) == self.m)
            assert (0.0 <= block_probs.min() <= block_probs.max() <= 1.0)
            assert (sum(block_probs) == 1.0)
            self.block_probs = block_probs
    def sample(self, N = None, block_memberships = None):
        """Samples an SBM using the basic (edge-by-edge) method. Optional block_memberships is a vector of memberships by block."""
        if (block_memberships is None):
            if (N is None):
                blocks_by_node = [i for i in range(self.m) for j in range(10)]  # 10 nodes per block by default
                N = 10 * self.m
            else:
                blocks_by_node = list(np.random.choice(self.m, N, p = self.block_probs))
        else:
            if (N is not None):
                assert (N == sum(block_memberships))
            blocks_by_node = [i for i in range(self.m) for j in range(block_memberships[i])]
        g = SBMGraph(self.m, blocks_by_node)
        for i in range(N):
            for j in range(i + 1, N):
                if (np.random.rand() <= self.edge_probs[blocks_by_node[i], blocks_by_node[j]]):
                    g.add_edge(i, j)
        return g

class SBMGraph(nx.Graph):
    """An instantiation of a Stochastic Block Model."""
    def __init__(self, m, blocks_by_node, *args, **kwargs):
        super(SBMGraph, self).__init__(*args, **kwargs)
        self.m = m
        self.N = len(blocks_by_node)
        self.blocks_by_node = blocks_by_node
        self.block_counts = np.bincount(self.blocks_by_node, minlength = self.m)
        if (min(self.block_counts) < 2):
            warnings.warn("Block exists with low membership (less than 2).")
        for i in range(self.N):
            self.add_node(i, block = self.blocks_by_node[i])
    def draw(self):
        """Assumes the nodes have been assigned blocks and colors them appropriately."""
        plt.clf()
        cmap = plt.cm.gist_ncar
        cdict = {i : cmap(int((i + 1) * cmap.N / (self.m + 1.0))) for i in range(self.m)}
        nx.draw_networkx(self, node_color = [cdict[self.blocks_by_node[i]] for i in range(self.N)], with_labels = False, node_size = 100) 
        plt.axes().get_xaxis().set_ticks([])
        plt.axes().get_yaxis().set_ticks([])
        plt.title("Stochastic Block Model\n%d blocks, %d nodes" % (self.m, self.N))
        plt.show(block = False)
    def empirical_block_probs(self, dtype = Fraction):
        safe_div = lambda x, y : (Fraction(x, y) if (dtype == Fraction) else (float(x) / y)) if (y != 0) else np.nan
        return np.vectorize(safe_div)(self.block_counts, self.N)
    def empirical_edge_probs(self, dtype = Fraction):
        nums = np.zeros((self.m, self.m), dtype = int)
        denoms = np.zeros((self.m, self.m), dtype = int)
        for i in range(self.m):
            for j in range(i, self.m):
                if (i == j):
                    denoms[i, i] = self.block_counts[i] * (self.block_counts[i] - 1) // 2
                else:
                    denoms[i, j] = denoms[j, i] = self.block_counts[i] * self.block_counts[j]
        for (v1, v2) in self.edges_iter():
            (i, j) = (self.blocks_by_node[v1], self.blocks_by_node[v2])
            nums[i, j] += 1
            if (i != j):
                nums[j, i] += 1
        safe_div = lambda x, y : (Fraction(x, y) if (dtype == Fraction) else (float(x) / y)) if (y != 0) else np.nan
        return np.vectorize(safe_div)(nums, denoms)

class SBM_MRF_Params(object):
    """Bundles together the random variables for an SBM graph."""
    def __init__(self, N, m):
        """Initialize from number of nodes and number of blocks in the model. A priori model puts uniform prior on each inter-block probability and a uniform Dirichlet prior on block membership probabilities. Each node's block membership then has categorical distribution based on these membership probabilities. Edges are treated automatically as observed (they are not random variables), and one Potential is assigned to the complete set of nodes and edge probabilities, based on the SBM joint probability."""
        self.N = N
        self.m = m
    def set_edge_probs(self, edge_probs = None, observed = False, prior = None):
        """Set the edge probabilities to a known matrix. prior is an m x m array of (alpha, beta) pairs for beta distribution priors (default (1, 1) uniform)."""
        if prior is None:
            alpha_betas = np.empty((self.m, self.m), dtype = object)
            for i in range(self.m):
                for j in range(i, self.m):
                    alpha_betas[i, j] = alpha_betas[j, i] = (1.0, 1.0)
        else:
            alpha_betas = prior
        if (edge_probs is not None):
            assert (edge_probs.shape == (self.m, self.m))
            assert np.allclose(edge_probs.transpose(), edge_probs)
            assert (0.0 <= edge_probs.min() <= edge_probs.max() <= 1.0)
        self.edge_probs = np.empty((self.m, self.m), dtype = object)
        for i in range(self.m):
            for j in range(i, self.m):
                if (edge_probs is None):
                    self.edge_probs[i, j] = self.edge_probs[j, i] = pymc.Beta('p%d_%d' % (i, j), alpha = alpha_betas[i, j][0], beta = alpha_betas[i, j][1])
                else:
                    self.edge_probs[i, j] = self.edge_probs[j, i] = pymc.Beta('p%d_%d' % (i, j), alpha = alpha_betas[i, j][0], beta = alpha_betas[i, j][1], value = edge_probs[i, j], observed = observed)
        self.edge_probs = pymc.Container(self.edge_probs)
    def set_block_probs(self, block_probs = None, observed = False, prior = None):
        """Set the block probabilities to a known vector. prior is Dirichlet prior hyperparameter theta (default uniform)."""
        theta = np.ones(self.m) if (prior is None) else prior
        if (block_probs is None):
            self._block_probs = pymc.Dirichlet('_block_probs', theta = theta)
        else:
            assert (len(block_probs) == self.m)
            assert (0.0 <= block_probs.min() <= block_probs.max() <= 1.0)
            assert (sum(block_probs) == 1.0)
            self._block_probs = pymc.Dirichlet('_block_probs', theta = theta, value = block_probs[:-1], observed = observed)
        self.block_probs = pymc.CompletedDirichlet('block_probs', self._block_probs)
    def set_blocks_by_node(self, blocks_by_node = dict(), observed = False):
        if isinstance(blocks_by_node, dict):
            self.blocks_by_node = np.empty(self.N, dtype = object)
            for i in range(self.N):
                # adopt convention that block is unknown if label is < 0
                if ((i in blocks_by_node) and (blocks_by_node[i] >= 0)):
                    self.blocks_by_node[i] = pymc.Categorical('b%d' % i, self.block_probs, value = blocks_by_node[i], observed = observed)
                else:
                    self.blocks_by_node[i] = pymc.Categorical('b%d' % i, self.block_probs)
            self.blocks_by_node = pymc.Container(self.blocks_by_node)
        else:
            self.set_blocks_by_node({node : block for (node, block) in enumerate(blocks_by_node)}, observed = observed)

class SBM_MRF(pymc.Model):
    def __init__(self, sbm_mrf_params):
        """PyMC model of an SBM. Initializes from an SBM_MRF_Params object."""
        for attr in ['N', 'm', 'edge_probs', '_block_probs', 'block_probs', 'blocks_by_node']:
            self.__dict__[attr] = sbm_mrf_params.__dict__[attr]
        super(SBM_MRF, self).__init__([self.edge_probs, self._block_probs, self.block_probs, self.blocks_by_node])
    def block_memberships(self, observed = True):
        """Returns vector of block memberships. -1 if observed = True and the node is not observed."""
        observation_set = self.observed_stochastics
        if (not observed):
            observation_set |= self.stochastics
        block_memberships = -1 * np.ones(self.N, dtype = float)
        for i in range(self.N):
            node = self.get_node('b%d' % i)
            if (node in observation_set):
                block_memberships[i] = int(node.value)
        return block_memberships
    def block_counts(self, observed = True):
        """Returns vector of observed block counts. Only counts observations if observed = True."""
        block_memberships = self.block_memberships(observed = observed)
        return np.bincount([block for block in block_memberships if (block >= 0)], minlength = self.m)
    def posterior_block_probs(self):
        """Based on the observed block memberships, returns the Dirichlet random variable for the posterior distribution of block probabilities as well as the completion of this random variable."""
        _posterior_block_probs = pymc.Dirichlet('_posterior_block_probs', theta = self._block_probs.parents['theta'] + self.observed_block_memberships())
        return (_posterior_block_probs, pymc.CompletedDirichlet('posterior_block_probs', _posterior_block_probs))
    def posterior_edge_probs(self):
        """Based on the edges or lack thereof between nodes with observed block memberships, returns the matrix of independent beta distributions for the posterior edge probabilities."""
        assert hasattr(self, 'graph')
        if (not hasattr(self, 'observed_subgraph')):
            observed_nodes = [i for i in range(self.N) if self.get_node('b%d' % i) in self.observed_stochastics]
            self.observed_subgraph = nx.Graph(self.graph).subgraph(observed_nodes)
        block_memberships = self.observed_block_memberships()
        pair_counts = defaultdict(int)
        for (i, j) in self.observed_subgraph.edges_iter():
            pair_counts


        nums = np.zeros((self.m, self.m), dtype = int)
        denoms = np.zeros((self.m, self.m), dtype = int)
        for i in range(self.m):
            for j in range(i, self.m):
                if (i == j):
                    denoms[i, i] = self.block_counts[i] * (self.block_counts[i] - 1) // 2
                else:
                    denoms[i, j] = denoms[j, i] = self.block_counts[i] * self.block_counts[j]
        for (v1, v2) in self.edges_iter():
            (i, j) = (self.blocks_by_node[v1], self.blocks_by_node[v2])
            nums[i, j] += 1
            if (i != j):
                nums[j, i] += 1
        safe_div = lambda x, y : (Fraction(x, y) if (dtype == Fraction) else (float(x) / y)) if (y != 0) else np.nan
        return np.vectorize(safe_div)(nums, denoms)


    @classmethod
    def from_sbm_graph(cls, graph):
        """Initializes MRF with SBMGraph. Sets the node block observations based on the graph's 'blocks_by_node' attribute."""
        params = SBM_MRF_Params(graph.N, graph.m)
        params.set_edge_probs()
        params.set_block_probs()
        params.set_blocks_by_node(graph.blocks_by_node, observed = True)
        mrf = cls(params)
        mrf.graph = graph
        @pymc.potential()
        def graph_potential(edge_probs = mrf.edge_probs, blocks_by_node = mrf.blocks_by_node):
            """Potential is product of probabilities of each node pair having or not having an edge in correspondence with the graph."""
            m = edge_probs.shape[0]
            N = len(blocks_by_node)
            block_counts = np.bincount(np.asarray(blocks_by_node, dtype = int), minlength = m)
            pair_counts = defaultdict(int)
            for (i, j) in mrf.graph.edges_iter():
                pair_counts[tuple(sorted((int(blocks_by_node[i]), int(blocks_by_node[j]))))] += 1
            logp = 0.0
            for i in range(m):
                for j in range(i, m):
                    lp_true = np.log(edge_probs[i, j])
                    lp_false = np.log(1 - edge_probs[i, j])
                    missing_edges = ((block_counts[i] * (block_counts[i] - 1) // 2) if (i == j) else (block_counts[i] * block_counts[j])) - pair_counts[(i, j)]
                    logp += pair_counts[(i, j)] * lp_true + missing_edges * lp_false
            return logp
        mrf.graph_potential = graph_potential
        mrf.potentials.add(mrf.graph_potential)
        return mrf


    

edge_probs = np.array([[0.1, 0.05], [0.05, 0.2]])
block_probs = np.array([0.7, 0.3])
s = SBM(edge_probs, block_probs)
g = s.sample(30)
mrf = SBM_MRF.from_sbm_graph(g)


