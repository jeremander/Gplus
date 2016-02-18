import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import pymc
import fractions
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
        self.n = edge_probs.shape[0]
        if (block_probs is None):
            self.block_probs = np.ones(self.n, dtype = float) / self.n  # uniform distribution
        else:
            assert (len(block_probs) == self.n)
            assert (0.0 <= block_probs.min() <= block_probs.max() <= 1.0)
            assert (sum(block_probs) == 1.0)
            self.block_probs = block_probs
    def sample(self, N = None, block_memberships = None):
        """Samples an SBM using the basic (edge-by-edge) method. Optional block_memberships is a vector of memberships by block."""
        if (block_memberships is None):
            if (N is None):
                blocks_by_node = [i for i in range(self.n) for j in range(10)]  # 10 nodes per block by default
                N = 10 * self.n
            else:
                blocks_by_node = list(np.random.choice(self.n, N, list(self.block_probs)))
        else:
            if (N is not None):
                assert (N == sum(block_memberships))
            blocks_by_node = [i for i in range(self.n) for j in range(block_memberships[i])]
        g = SBMGraph(self.n)
        for i in range(N):
            g.add_node(i, block = blocks[i])
        for i in range(N):
            for j in range(i + 1, N):
                if (np.random.rand() <= self.edge_probs[blocks_by_node[i], blocks_by_node[j]]):
                    g.add_edge(i, j)
        return g

class SBMGraph(nx.Graph):
    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.N = self.number_of_nodes()
        self.blocks_by_node = [self.node[i]['block'] for i in range(self.N)]
    def draw(self):
        """Assumes the nodes have been assigned blocks and colors them appropriately."""
        plt.clf()
        cmap = plt.cm.gist_ncar
        cdict = {i : cmap(int((i + 1) * cmap.N / (self.n + 1.0))) for i in range(self.n)}
        nx.draw_networkx(self, node_color = [cdict[self.blocks_by_node[i]] for i in range(self.N)], with_labels = False, node_size = 200) 
        plt.axes().get_xaxis().set_ticks([])
        plt.axes().get_yaxis().set_ticks([])
        plt.title("Stochastic Block Model\n%d blocks, %d nodes" % (self.n, self.N))
        plt.show()
    def empirical_block_probs(self):
        return 



