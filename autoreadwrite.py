import pandas as pd
import time
import pickle
import igraph as ig
import networkx as nx

def time_format(seconds):
    """Formats a time into a convenient string."""
    s = ''
    if (seconds >= 3600):
        s += "%dh," % (seconds // 3600)
    if (seconds >= 60):
        s += "%dm," % ((seconds % 3600) // 60)
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

def load_object(folder, obj_name, extension):
    filename = folder + '/' + obj_name + '.' + extension
    did_load = False
    try:
        print("\nLoading %s from '%s'..." % (obj_name, filename))
        if (extension == 'csv'):
            obj = timeit(pd.read_csv)(filename)
            did_load = True
        elif (extension == 'pickle'):
            obj = timeit(pickle.load)(open(filename, 'rb'))
            did_load = True
        elif (extension == 'ig.edges'):
            obj = timeit(ig.Graph.Read_Edgelist)(filename, directed = False)
            did_load = True
        elif (extension == 'nx.edges'):
            obj = timeit(nx.read_edgelist)(filename)
            did_load = True
        if did_load:
            print("Successfully loaded %s." % obj_name)
            return obj
    except:
        pass
    if (not did_load):
        raise IOError("Could not load %s from file." % obj_name)

def save_object(obj, folder, obj_name, extension):
    filename = folder + '/' + obj_name + '.' + extension
    did_save = False
    try:
        print("\nSaving %s to '%s'..." % (obj_name, filename))
        if (extension == 'csv'):
            timeit(pd.DataFrame.to_csv)(obj, filename, index = False)
            did_save = True
        elif (extension == 'pickle'):
            timeit(pickle.dump)(obj, open(filename, 'wb'))
            did_save = True
        elif (extension == 'ig.edges'):
            timeit(ig.Graph.write_edgelist)(obj, filename)
            did_save = True
        elif (extension == 'nx.edges'):
            timeit(nx.write_edgelist)(obj, filename, data = False)
            did_save = True
        if did_save:
            print("Successfully saved %s." % obj_name)
    except:
        pass
    if (not did_save):
        raise IOError("Failed to save %s to file." % obj_name)

def autoreadwrite(obj_names, extensions):
    """Decorator for methods that construct member objects. Initially tries to load each member object from a file in the object's 'folder' member, whose filename is the constructed member object with a .csv, .pickle, or .edges extension, depending on the format. return_obj is the name of the object to return, if any."""
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
            did_load_flags = []
            for obj_name, extension in zip(obj_names, extensions):
                try:
                    self.__dict__[obj_name] = load_object(self.folder, obj_name, extension)
                except:
                    print("Could not load %s from file.\n\nConstructing from scratch..." % obj_name)
                    timeit(func)(*args, **kwargs)
                    break
            if save:
                for obj_name, extension in zip(obj_names, extensions):
                    save_object(self.__dict__[obj_name], self.folder, obj_name, extension)
        return autoreadwrite_wrapped_function
    return autoreadwrite_decorator


class ObjectWithReadwriteProperties(object):
    """Class that may contain properties loaded from a file. The properties are specified in the class's 'readwrite_properties' member dictionary, which maps properties to file extensions for saving/loading. Each instance of the class must also have a 'folder' member indicating where to save/load the data. The convention is that if a property is already possessed by the instance, accessing it will return the object as usual. If it is not possessed by the instance, it will attempt to load it from a file using the appropriate path, then return it. The 'load_[property]' method will load the property from a file but not return it, and the 'save_[property]' method will save the property to a file."""
    readwrite_properties = dict()
    def __init__(self, folder):
        self.folder = folder
        def make_property(property_name):
            hidden_property_name = '_' + property_name
            def property_load():
                print("In property_load")
                if (property_name not in self.__class__.readwrite_properties):
                    raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, property_name))
                if (not hasattr(self, hidden_property_name)):
                    self.__dict__[hidden_property_name] = load_object(self.folder, property_name, self.__class__.readwrite_properties[property_name])
            def property_get():
                print("In property_get")
                property_load()
                return self.__dict__[hidden_property_name]
            def property_save():
                print("In property_save")
                if (not hasattr(self, hidden_property_name)):
                    raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, property_name))
                save_object(self.__dict__[hidden_property_name], self.folder, property_name, self.__class__.readwrite_properties[property_name])
            return (property_load, property_get, property_save)
        for property_name in self.__class__.readwrite_properties:
            (loader, getter, saver) = make_property(property_name)
            self.__dict__['load_' + property_name] = loader
            self.__dict__['get_' + property_name] = getter
            self.__dict__['save_' + property_name] = saver
    def __getattr__(self, attr):
        print("In getattribute")
        if (attr in self.__class__.readwrite_properties):
            return self.__dict__['get_' + attr]()
        return self.__getattribute__(attr)
    def __delattr__(self, attr):
        if (attr in self.__class__.readwrite_properties):
            super().__delattr__('_' + attr)
        else:
            super().__delattr__(attr)
