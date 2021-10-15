
from copy import deepcopy
from functools import reduce
from math import ceil, floor
from obspy import UTCDateTime
from os.path import abspath, join
from retry import retry

import copy
import csv
import json
import time
import numpy as np
import obspy
import re
import uuid
import warnings
import zipfile


try:
    from urllib import URLopener
except ImportError:
    from urllib.request import URLopener


class AttribDict(obspy.core.util.attribdict.AttribDict):
    pass


def asarray(x):
    """ Numpy array typecast
    """
    return np.array(x, dtype=np.float64, ndmin=1, copy=False)


def is_mpi_env():
    try:
        import mpi4py
    except ImportError:
        return False

    try:
        import mpi4py.MPI
    except ImportError:
        return False

    if mpi4py.MPI.COMM_WORLD.Get_size()>1:
        return True
    else:
        return False


def iterable(arg):
    """ Simple list typecast
    """
    from mtuq.grid import Grid, UnstructuredGrid
    if not isinstance(arg, (list, tuple, Grid, UnstructuredGrid, np.ndarray)):
        return [arg]
    else:
        return arg


def merge_dicts(*dicts):
   merged = {}
   for dict in dicts:
      merged.update(dict)
   return merged


def product(*arrays):
    return reduce((lambda x, y: x * y), arrays)


def remove_list(list1, list2):
    """ Removes all items of list2 from list1
    """
    for item in list2:
        try:
            list1.remove(item)
        except ValueError:
            pass
    return list1


def replace(string, *args):
    narg = len(args)

    iarg = 0
    while iarg < narg:
        string = re.sub(args[iarg], args[iarg+1], string)
        iarg += 2
    return string


def timer(func):
    """ Decorator for measuring execution time; prints elapsed time to
        standard output
    """
    def timed_func(*args, **kwargs):
        start_time = time.time()

        output = func(*args, **kwargs)

        if kwargs.get('verbose', True):
            _elapsed_time = time.time() - start_time
            print('  Elapsed time (s): %f\n' % _elapsed_time)

        return output

    return timed_func


def basepath():
    """ MTUQ base directory
    """
    import mtuq
    return abspath(join(mtuq.__path__[0], '..'))


def fullpath(*args):
    """ Prepends MTUQ base diretory to given path
    """
    return join(basepath(), *args)


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):

        from mtuq import Station
        if isinstance(obj, Station):
            # don't write out SAC metadata (too big)
            if hasattr(obj, 'sac'):
                obj = deepcopy(obj)
                obj.pop('sac', None)

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, AttribDict):
            return obj.__dict__
        if issubclass(type(obj), AttribDict):
            return obj.__dict__
        if isinstance(obj, UTCDateTime):
            return str(obj)

        return super(JSONEncoder, self).default(obj)


def save_json(filename, data):
    if type(data) == AttribDict:
        data = {key: data[key] for key in data}

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, cls=JSONEncoder, ensure_ascii=False, indent=4)


def timer(func):
    """ Decorator for measuring execution time
    """
    def timed_func(*args, **kwargs):
        if kwargs.get('timed', True):
            start_time = time.time()
            output = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            print('  Elapsed time (s): %f\n' % elapsed_time)
            return output
        else:
            return func(*args, **kwargs)

    return timed_func


def unzip(filename):
    parts = filename.split('.')
    if parts[-1]=='zip':
        dirname = '.'.join(parts[:-1])
    else:
        dirname = filename
        filename += '.zip'

    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(dirname)
    zip_ref.close()

    return dirname


def warn(*args, **kwargs):
    if is_mpi_env():
        from mpi4py import MPI
        if MPI.COMM_WORLD.rank==0:
           warnings.warn(*args, **kwargs)
    else:
       warnings.warn(*args, **kwargs)


@retry(Exception, tries=4, delay=2, backoff=2)
def urlopen_with_retry(url, filename):
    opener = URLopener()
    opener.retrieve(url, filename)


def url2uuid(url):
    """ Converts a url to a uuid string
    """
    namespace = uuid.NAMESPACE_URL
    name = url
    return uuid.uuid5(namespace, name)


class Null(object):
    """ Always and reliably does nothing
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __nonzero__(self):
        return False


class ProgressCallback(object):
    """ Displays progress messages

    Displays messages at a regular interval (specified as a percentagage of the
    total number of iterations), when called from a loop or other iterative
    procedure
    """
    def __init__(self, start, stop, percent):

        start = int(round(start))
        stop = int(round(stop))

        assert (0 <= start)
        assert (start <= stop)

        if percent==0.:
            self.iter = start
            self.next_iter = float("inf")
            return
        elif 0 < percent <= 1:
            percent = 1
        elif 1 < percent <= 50.:
            percent = int(ceil(percent))
        elif 50 < percent <= 100:
            percent = 50
        else:
            raise ValueError

        self.start = start
        self.stop = stop
        self.percent = percent
        self.msg_interval = percent/100.*stop
        self.msg_count = int(100./percent*start/stop)
        self.iter = start
        self.next_iter = self.msg_count * self.msg_interval


    def __call__(self):
        if self.iter >= self.next_iter:
            print("  about %d percent finished" % (self.msg_count*self.percent))
            self.msg_count += 1
            self.next_iter = self.msg_count * self.msg_interval
        self.iter += 1


def dataarray_idxmin(da):
    """ idxmin helper function
    """
    # something similar to this has now been implemented in a beta version
    # of xarray
    da = da.where(da==da.min(), drop=True).squeeze()
    if da.size > 1:
        warn("No unique global minimum\n")
        return da[0].coords
    else:
        return da.coords


def dataarray_idxmax(da):
    """ idxmax helper function
    """
    # something similar to this has now been implemented in a beta version
    # of xarray
    da = da.where(da==da.max(), drop=True).squeeze()
    if da.size > 1:
        warn("No unique global maximum\n")
        return da[0].coords
    else:
        return da.coords
