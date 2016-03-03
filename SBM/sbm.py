import fractions
import warnings
import scipy.stats
from collections import defaultdict
from vngraph import *
from importlib import reload


def plot_beta_pdf(rv):
    """Plot a beta distribution pdf. Input is an instance of scipy.stats.beta."""
    xs = np.linspace(0.0, 1.0, 10000)
    plt.plot(xs, rv.pdf(xs), linewidth = 2)
    plt.title("Beta pdf\nalpha = %.3f, beta = %.3f" % (rv.args[0], rv.args[1]))
    plt.show(block = False)

class Pair(object):
    def __init__(self, first, second):
        self.items = (first, second)
    def __getitem__(self, i):
        return self.items[i]
    def __repr__(self):
        return str(self.items)

class Fraction(fractions.Fraction):
    def __repr__(self):
        return str(self)

def safe_div(x, y, dtype = float):
    if (dtype == Pair):
        return Pair(x, y)
    elif (dtype == Fraction):
        return Fraction(x, y)
    else:
        return (float(x) / y) if (y != 0) else np.nan

def empirical_block_probs(blocks_by_node, observed_flags, m, dtype = float):
    """Computes empirical block membership probabilities (known only)."""
    block_counts = np.bincount(blocks_by_node[observed_flags], minlength = m)    
    return np.vectorize(lambda x, y : safe_div(x, y, dtype))(block_counts, block_counts.sum())

def empirical_edge_probs(blocks_by_node, observed_flags, m, edges_iter, dtype = float):
    """Computes empirical edge probabilities (between pairs of known edges only). edges_iter is an iterator of edges."""
    block_counts = np.bincount(blocks_by_node[observed_flags], minlength = m)    
    nums = np.zeros((m, m), dtype = int)
    denoms = np.zeros((m, m), dtype = int)
    for i in range(m):
        for j in range(i, m):
            if (i == j):
                denoms[i, i] = block_counts[i] * (block_counts[i] - 1) // 2
            else:
                denoms[i, j] = denoms[j, i] = block_counts[i] * block_counts[j]
    for (v1, v2) in edges_iter:
        (i, j) = (blocks_by_node[v1], blocks_by_node[v2])
        if (observed_flags[v1] and observed_flags[v2]):
            nums[i, j] += 1
            if (i != j):
                nums[j, i] += 1
    return np.vectorize(lambda x, y : safe_div(x, y, dtype))(nums, denoms)

class SBM(object):
    """Class for Stochastic Block Models."""
    def __init__(self, edge_probs, block_probs = None):
        """edge_probs is symmetric matrix of communication probabilities; block_probs is vector of block probabilities."""
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
        g = SBMGraph(self.m, np.array(blocks_by_node))
        for i in range(N):
            for j in range(i + 1, N):
                if (np.random.rand() <= self.edge_probs[blocks_by_node[i], blocks_by_node[j]]):
                    g.add_edge(i, j)
        return g

class TwoBlockSBM(SBM):
    """Special case of SBM where there are only two blocks."""
    def __init__(self, p0_0, p0_1, p1_1, p1 = None):
        edge_probs = np.array([[p0_0, p0_1], [p0_1, p1_1]])
        block_probs = None if (p1 is None) else np.array([1.0 - p1, p1])
        super().__init__(edge_probs, block_probs)

class SBMGraph(VNGraph):
    """An instantiation of a Stochastic Block Model."""
    def __init__(self, m, blocks_by_node, observed_flags = None):
        super().__init__()
        self.m = m
        self.N = len(blocks_by_node)
        self.blocks_by_node = blocks_by_node
        self.observed_flags = np.ones(self.N, dtype = bool) if (observed_flags is None) else observed_flags
        # only include known blocks
        self.block_counts = np.bincount(self.blocks_by_node[self.observed_flags], minlength = self.m)
        if (min(self.block_counts) < 2):
            warnings.warn("Block exists with low membership (less than 2).")
        for i in range(self.N):
            self.add_node(i, block = self.blocks_by_node[i])
    def mcar_occlude(self, num_to_occlude = None, occlusion_prob = 0.75):
        """Occludes nodes in an i.i.d. Bernoulli fashion. Either the probability of occlusion is supplied or the number to occlude is supplied."""
        if (num_to_occlude is None):
            occlusion_flags = (np.random.rand(self.N) <= occlusion_prob)
        else:
            assert (num_to_occlude <= self.N)
            occlusion_flags = np.random.permutation(([True] * num_to_occlude) + ([False] * (self.N - num_to_occlude)))
        g = SBMGraph(self.m, self.blocks_by_node, ~occlusion_flags)
        for (i, j) in self.edges_iter():
            g.add_edge(i, j)
        return g
    def to_two_block_sbm(self, selected_block):
        """Converts the SBMGraph into an identical graph with only two blocks, where the selected block becomes block 1, and the rest of the blocks are combined to form block 0."""
        g = SBMGraph(self.m, np.array([1 if (block == selected_block) else 0]), self.observed_flags)
        for (i, j) in self.edges_iter():
            g.add_edge(i, j)
        return g
    def draw(self):
        """Assumes the nodes have been assigned blocks and colors them appropriately. Unobserved nodes are colored black."""
        plt.clf()
        cmap = plt.cm.gist_ncar
        cdict = {i : cmap(int((i + 1) * cmap.N / (self.m + 1.0))) for i in range(self.m)}
        black_color = (0.0, 0.0, 0.0, 1.0)
        nx.draw_networkx(self, node_color = [(cdict[self.blocks_by_node[i]] if self.observed_flags[i] else black_color) for i in range(self.N)], with_labels = False, node_size = 100) 
        plt.axes().get_xaxis().set_ticks([])
        plt.axes().get_yaxis().set_ticks([])
        plt.title("Stochastic Block Model\n%d blocks, %d nodes" % (self.m, self.N))
        plt.show(block = False)
    def empirical_block_probs(self, dtype = float):
        return empirical_block_probs(self.blocks_by_node, self.oberved_flags, self.m, dtype = dtype)
    def empirical_edge_probs(self, dtype = float):
        return empirical_edge_probs(self.blocks_by_node, self.observed_flags, self.m, self.edges_iter(), dtype = dtype)



class MCMC(object):
    def plot(self, var_name, true_dist = None):
        """Presents a trace plot, autocorrelation plot, and sample frequency histogram (normalized) for a given variable name to aid in visual inspection of MCMC convergence. If true_dist is supplied, plots this pdf on top of the histogram."""
        samps = self.traces[var_name]
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax1.plot(samps)
        ax1.set_title('Trace plot of %s' % name)
        ax1.set_xlabel('sample #')
        ax2 = fig.add_subplot(312)
        ax2.acorr(samps - samps.mean(), maxlags = 100)
        ax2.set_title('Autocorrelations')
        ax2.set_xlabel('lag')
        ax2.set_ylabel('corr')
        ax2.set_xlim((0, 100))
        ax3 = fig.add_subplot(313)
        ax3.hist(samps, bins = 100, normed = True, color = 'blue')
        if (true_dist is not None):
            xs = np.linspace(samps.min(), samps.max(), 1000)
            ax3.plot(xs, true_dist.pdf(xs), color = 'red', linewidth = 2)
        ax3.set_title('Empirical PDF')
        plt.tight_layout()
        plt.show(block = False)

class SBM_MRF(MCMC):
    """Markov Random Field model of an SBM where parameters are unknown, but where all edges and possibly some node memberships are observed. The parameters are given prior distributions (Dirichlet/beta for block probabilities, i.i.d. beta for edge probabilities). This model is capable of performing Gibbs sampling to estimate the posterior distributions of parameters and unknown node memberships."""
    def __init__(self, graph, theta = None, alphas = None, betas = None):
        """Initialize with SBMGraph (where -1's represent unknown nodes). If vector theta is supplied, this is hyperparameter vector (dimension m) for Dirichlet prior of block probabilities. If alpha and betas are supplied, they are each m x m arrays giving the beta distribution hyperparameters for the edge probabilities. Default is uniform alpha = 1, beta = 1."""
        self.graph = graph
        self.N, self.m = self.graph.N, self.graph.m
        if (theta is None):
            self.theta = np.ones(self.m)
        else:
            assert (len(theta) == self.m)
            self.theta = theta
        if (alphas is None):
            self.alphas = np.ones((self.m, self.m), dtype = float)
        else:
            assert ((alphas.shape == (self.m, self.m)) and (alphas.transpose() == alphas).all())
            self.alphas = alphas
        if (betas is None):
            self.betas = np.ones((self.m, self.m), dtype = float)
        else:
            assert ((betas.shape == (self.m, self.m)) and (betas.transpose() == betas).all())
            self.betas = betas
        # prior Dirichlet prior distribution (beta if m = 2)
        self.block_prob_prior = scipy.stats.dirichlet(self.theta) if (self.m > 2) else scipy.stats.beta(self.theta[1], self.theta[0])
        # array of prior beta prior distributions
        self.edge_prob_priors = np.vectorize(lambda alpha, beta : scipy.stats.beta(alpha, beta))(self.alphas, self.betas)
        # a copy of the block memberships that will change state
        self.blocks_by_node = np.array(self.graph.blocks_by_node)  
    def set_posteriors(self, observed = True):
        """Sets the posterior distributions (Dirichlet & betas) based on the current block memberships. If observed = True, only includes observed nodes in forming the posteriors."""
        observed_flags = self.graph.observed_flags if observed else np.ones(self.N)
        block_probs = empirical_block_probs(self.blocks_by_node, observed_flags, self.m, dtype = Pair)
        edge_probs = empirical_edge_probs(self.blocks_by_node, observed_flags, self.m, self.graph.edges_iter(), dtype = Pair)
        # Dirichlet conjugate prior -> Dirichlet posterior
        posterior_theta = self.theta + np.array([pair[0] for pair in block_probs], dtype = float)
        self.block_prob_posterior = scipy.stats.dirichlet(posterior_theta) if (self.m > 2) else scipy.stats.beta(posterior_theta[1], posterior_theta[0])
        # beta conjugate prior -> beta posterior
        self.edge_prob_posteriors = np.empty((self.m, self.m), dtype = object)
        for i in range(self.m):
            for j in range(self.m):
                self.edge_prob_posteriors[i, j] = scipy.stats.beta(self.alphas[i, j] + edge_probs[i, j][0], self.betas[i, j] + edge_probs[i, j][1])
    def set_sbm(self, sbm):
        """Sets the current state of the SBM parameters with an SBM object."""
        assert (isinstance(sbm, SBM) and (sbm.m == self.m))
        self.sbm = sbm
    def init_sbm(self, style = 'MAP'):
        """Initializes the SBM parameters for MCMC. If style is 'MAP', uses the empirical counts to initialize these parameters with the MAP estimate; if style is 'prior', use the prior distributions; if style is 'posterior', use the posterior distributions."""
        if (style == 'MAP'):
            edge_probs = empirical_edge_probs(self.graph.blocks_by_node, self.graph.observed_flags, self.m, self.graph.edges_iter(), dtype = float)
            block_probs = empirical_block_probs(self.graph.blocks_by_node, self.graph.observed_flags, self.m, dtype = float)
        elif (style == 'prior'):
            edge_probs = np.zeros((self.m, self.m), dtype = float)
            for i in range(self.m):
                for j in range(i, self.m):
                    edge_probs[i, j] = edge_probs[j, i] = self.edge_prob_priors[i, j].rvs()
            block_probs = self.block_prob_prior.rvs()[0] if (self.m > 2) else self.block_prob_prior.rvs()
        else:  # posterior
            self.set_posteriors(observed = True)  # first time means there should be unobserved nodes
            edge_probs = np.zeros((self.m, self.m), dtype = float)
            for i in range(self.m):
                for j in range(i, self.m):
                    edge_probs[i, j] = edge_probs[j, i] = self.edge_prob_posteriors[i, j].rvs()
            block_probs = self.block_prob_posterior.rvs()[0] if (self.m > 2) else self.block_prob_posetior.rvs()
        self.set_sbm(SBM(edge_probs, block_probs))
    def init_unobserved_nodes(self):
        """After the SBM parameters have been set, initialize the unobserved node memberships as i.i.d. samples from the SBM block probability distribution (note: this ignores conditioning on the edges). The purpose of this is just to give a random starting point for the Gibbs sampling."""
        cdf = self.sbm.block_probs.cumsum()
        for i in range(self.N):
            if (not self.graph.observed_flags[i]):  # sample from categorical distribution
                self.blocks_by_node[i] = cdf.searchsorted(np.random.rand())

    # @pymc.potential()
    # def graph_potential(edge_probs = mrf.edge_probs, blocks_by_node = mrf.blocks_by_node):
    #     """Potential is product of probabilities of each node pair having or not having an edge in correspondence with the graph."""
    #     m = edge_probs.shape[0]
    #     N = len(blocks_by_node)
    #     block_counts = np.bincount(np.asarray(blocks_by_node, dtype = int), minlength = m)
    #     pair_counts = defaultdict(int)
    #     for (i, j) in mrf.graph.edges_iter():
    #         pair_counts[tuple(sorted((int(blocks_by_node[i]), int(blocks_by_node[j]))))] += 1
    #     logp = 0.0
    #     for i in range(m):
    #         for j in range(i, m):
    #             lp_true = np.log(edge_probs[i, j])
    #             lp_false = np.log(1 - edge_probs[i, j])
    #             missing_edges = ((block_counts[i] * (block_counts[i] - 1) // 2) if (i == j) else (block_counts[i] * block_counts[j])) - pair_counts[(i, j)]
    #             logp += pair_counts[(i, j)] * lp_true + missing_edges * lp_false
    #     return logp
    # mrf.graph_potential = graph_potential
    # mrf.potentials.add(mrf.graph_potential)
    # return mrf

s1 = TwoBlockSBM(0.1, 0.05, 0.2, p1 = 0.25)
g1 = s1.sample(100)
mrf1 = SBM_MRF(g1)
s2 = TwoBlockSBM(0.1, 0.05, 0.1, p1 = 0.25)
g2 = s2.sample(100)
mrf2 = SBM_MRF(g2)
s3 = TwoBlockSBM(0.1, 0.05, 0.2, p1 = 0.5)
g3 = s3.sample(100)
mrf3 = SBM_MRF(g3)
s4 = TwoBlockSBM(0.1, 0.05, 0.1, p1 = 0.5)  # non-identifiable
g4 = s4.sample(100)
mrf4 = SBM_MRF(g4)

def main():
    pass



if __name__ == "__main__":
    main()

