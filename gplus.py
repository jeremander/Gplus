import pandas as pd
import numpy as np
import igraph as ig
import louvain
from importlib import reload
from autoreadwrite import *


class Gplus(ig.Graph):
    """Graph of Google+ data."""
    @autoreadwrite(['degree_dict', 'degree_power_law'], ['pickle', 'pickle'])
    def degrees(self, save = False):
        """Creates dictionary of node degrees."""
        self.degree_dict = dict((v.index, v.degree()) for v in self.vs)
        self.degree_power_law = ig.power_law_fit(list(self.degree_dict.values()))
    @autoreadwrite(['comp_sizes'], ['csv'])
    def component_sizes(self, save = False):
        """Computes connected component sizes."""
        components = self.components()
        self.comp_sizes = pd.DataFrame([len(components.subgraph(i)) for i in range(len(components))], columns = ['componentSize'])
    @autoreadwrite(['fastgreedy_comms'], ['pickle'])
    def fastgreedy_communities(self, save = False):
        """Computes communities obtained by iGraph's fastgreedy algorithm (approximately linear time)."""
        self.fastgreedy_comms = self.community_fastgreedy()
    @autoreadwrite(['louvain_memberships'], ['csv'])
    def louvain(self, save = False):
        """Computes cluster memberships returned by the Louvain method (implemented in C++ via louvain-igraph package)."""
        self.louvain_memberships = pd.DataFrame(louvain.find_partition(self, method = 'Modularity').membership, columns = ['louvainMembership'])
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



