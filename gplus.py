import pandas as pd
import numpy as np
import igraph as ig
import pickle
import time
from importlib import reload

def time_format(seconds):
    """Formats a time into a convenient string."""
    s = ''
    if (seconds >= 3600):
        s += "%dh," % (seconds // 3600)
    if (seconds >= 60):
        s += "%dm," % (seconds // 60)
    s += "%.3fs" % (seconds % 60)
    return s

def timeit(func):
    """Decorator for annotating function call with elapsed time info."""
    def timed(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(time_format(elapsed_time))
        return result
    return timed

def autoreadwrite(obj_names, extensions):
    """Decorator for methods that construct member objects. Initially tries to load each member object from a file in the object's 'folder' member, whose filename is the constructed member object with either a .csv or .pickle extension, depending on the format."""
    def autoreadwrite_decorator(func):
        def autoreadwrite_wrapped_function(*args, **kwargs):
            self = args[0]
            argnames = list(func.__code__.co_varnames)
            if ('save' in argnames[:len(args)]):
                save = args[argnames.index('save')]
            elif ('save' in kwargs):
                save = kwargs['save']
            else:
                save = False
            for obj_name, extension in zip(obj_names, extensions):
                filename = self.folder + '/' + obj_name + '.' + extension
                if (not hasattr(self, obj_name)):
                    did_load = False
                    try:
                        print("\nLoading %s from '%s'..." % (obj_name, filename))
                        start_time = time.time()
                        if (extension == 'csv'):
                            self.__dict__[obj_name] = timeit(pd.read_csv)(filename)
                            did_load = True
                        elif (extension == 'pickle'):
                            self.__dict__[obj_name] = timeit(pickle.load)(open(filename, 'rb'))
                            did_load = True
                        if hasattr(self, obj_name):
                            print("Successfully loaded %s." % obj_name)
                    except:
                        pass
                    if (not did_load):
                        print("Could not load %s from file." % obj_name)
                        timeit(func)(*args, **kwargs)
                if save:
                    did_save = False
                    try:
                        print("Saving %s to '%s'..." % (obj_name, filename))
                        if (extension == 'csv'):
                            timeit(pd.DataFrame.to_csv)(self.__dict__[obj_name], filename, index = False)
                            did_save = True
                        elif (extension == 'pickle'):
                            timeit(pickle.dump)(self.__dict__[obj_name], open(filename, 'wb'))
                            did_save = True
                    except:
                        pass
                    if did_save:
                        print("Successfully saved %s." % obj_name)
                    else:
                        print("Failed to save %s to file." % obj_name)
            return self.__dict__[obj_names[0]]  # return only the first object (others are stored)
        return autoreadwrite_wrapped_function
    return autoreadwrite_decorator


class Gplus(ig.Graph):
    """Graph of Google+ data."""
    @autoreadwrite(['degree_dict', 'degree_power_law'], ['pickle', 'pickle'])
    def degrees(self, save = False):
        """Returns dictionary of node degrees."""
        self.degree_dict = dict((v.index, v.degree()) for v in self.vs)
        self.degree_power_law = ig.power_law_fit(list(self.degree_dict.values()))
    @autoreadwrite(['comp_sizes'], ['csv'])
    def component_sizes(self, save = False):
        """Returns connected component sizes."""
        components = self.components()
        self.comp_sizes = pd.DataFrame(np.array([len(components.subgraph(i)) for i in range(len(components))]), columns = ['componentSize'])
    @autoreadwrite(['fastgreedy_comms'], ['pickle'])
    def fastgreedy_communities(self, save = False):
        """Returns communities obtained by iGraph's fastgreedy algorithm (approximately linear time)."""
        self.fastgreedy_comms = self.community_fastgreedy()
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



