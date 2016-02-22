import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import pymc
import fractions
import warnings
from importlib import reload

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
                blocks_by_node = list(np.random.choice(self.m, N, list(self.block_probs)))
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
        super().__init__(*args, **kwargs)
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
        nx.draw_networkx(self, node_color = [cdict[self.blocks_by_node[i]] for i in range(self.N)], with_labels = False, node_size = 200) 
        plt.axes().get_xaxis().set_ticks([])
        plt.axes().get_yaxis().set_ticks([])
        plt.title("Stochastic Block Model\n%d blocks, %d nodes" % (self.m, self.N))
        plt.show(block = False)
    def empirical_membership_probs(self, dtype = Fraction):
        safe_div = lambda x, y : (Fraction(x, y) if (dtype == Fraction) else (x / y)) if (y != 0) else np.nan
        return np.vectorize(safe_div)(self.block_counts, self.N)
    def empirical_block_probs(self, dtype = Fraction):
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
        safe_div = lambda x, y : (Fraction(x, y) if (dtype == Fraction) else (x / y)) if (y != 0) else np.nan
        return np.vectorize(safe_div)(nums, denoms)

class SBMModel():
    pass

