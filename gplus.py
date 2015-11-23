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
        s += "%2dm," % (seconds // 60)
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

def autoreadwrite(obj_name, extension):
    """Decorator for methods that construct a member object. Initially tries to load the member object from a file in the object's 'folder' member, whose filename is the constructed member object with either a .csv or .pickle extension, depending on the format."""
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
            filename = self.folder + '/' + obj_name + '.' + extension
            if (not hasattr(self, obj_name)):
                did_load = False
                try:
                    print("Loading data from '%s'..." % filename)
                    start_time = time.time()
                    if (extension == 'csv'):
                        self.__dict__[obj_name] = timeit(pd.read_csv)(filename, index = False)
                        did_load = True
                    elif (extension == 'pickle'):
                        self.__dict__[obj_name] = timeit(pickle.load)(open(filename, 'rb'))
                        did_load = True
                    if hasattr(self, obj_name):
                        print("Successfully loaded data.")
                except:
                    pass
                if (not did_load):
                    print("Could not load object from file.")
                    timeit(func)(*args, **kwargs)
            if save:
                did_save = False
                try:
                    print("Saving data to '%s'..." % filename)
                    if (extension == 'csv'):
                        timeit(pd.DataFrame.to_csv)(self.__dict__[obj_name], filename, index = False)
                        did_save = True
                    elif (extension == 'pickle'):
                        timeit(pickle.dump)(self.__dict__[obj_name], open(filename, 'wb'))
                        did_save = True
                except:
                    pass
                if did_save:
                    print("Successfully saved data.")
                else:
                    print("Failed to save object to file.")
            return self.__dict__[obj_name]
        return autoreadwrite_wrapped_function
    return autoreadwrite_decorator


class Gplus(ig.Graph):
    """Graph of Google+ data."""
    @autoreadwrite('degree_dict', 'pickle')
    def degrees(self, save = False):
        """Returns dictionary of node degrees."""
        self.degree_dict = dict((v.index, v.degree()) for v in self.vs)
    @autoreadwrite('comp_sizes', 'csv')
    def component_sizes(self, save = False):
        """Returns connected component sizes."""
        components = self.components()
        self.comp_sizes = pd.DataFrame(np.array([len(components.subgraph(i)) for i in range(len(components))]), columns = ['componentSize'])
    @autoreadwrite('fastgreedy_comms', 'pickle')
    def fastgreedy_communities(self, save = False):
        """Returns communities obtained by iGraph's fastgreedy algorithm (approximately linear time)."""
        self.fastgreedy_comms = self.community_fastgreedy()
    def __len__(self):
        return len(self.vs)
    @classmethod 
    def from_data(cls, folder = 'gplus0/data', directed = False):
        """Reads the graph from edge list."""
        filename = folder + '/directed_edges.dat'
        print("Loading data from '%s'..." % filename)
        g = timeit(cls.Read_Edgelist)(filename)
        print("Successfully loaded data.")
        if (not directed):
            print("Removing directions from edges...")
            timeit(Gplus.to_undirected)(g)
        g.folder = folder
        return g
