import os
import re
from datetime import datetime
import numpy as np


def get_dt_stamp():
    """
    Get date-timestamp.

    Returns
    -------
    str
        Date-timestamp in string format separated by '_'.
    """
    return re.split(
        '\.', str(datetime.now()))[0].replace(
        ' ',
        '_').replace(
        ':',
        '_').replace(
        '-',
        '_')


def flatten_list(l, log):
    """
    Flatten a list of lists.

    Parameters
    ----------
    l : list of lists
    log
        Log for warning messages.
    """
    warning_msg = 'Warning: returning None.'
    if l is None or l[0] is None:
        if log is not None:
            log.info(warning_msg)
        else:
            print warning_msg
        return [None]
    else:
        return [val for sublist in l for val in sublist]


def import_module(dataset, model_dir='dataset_processing'):
    """
    Dynamically import a module.

    Parameters
    ----------
    dataset : string
        Name of dataset to be encoded. Dataset should be found in
        /dataset_processing and have its own 'init' function.
    """
    return getattr(
        __import__(model_dir, fromlist=[dataset]), dataset)


def make_dir(d):
    """
    Make directory d if it does not exist.

    Parameters
    ----------
    d : string
        Name of directory to be created.
    """
    if not os.path.exists(d):
        os.makedirs(d)


def save_npys(data, model_name, output_string):
    """
    Save key/values in data as numpys.

    Parameters
    ----------
    data : dictionary
        Dictionary containing data.
    model_name: string
        Name of model that was used to create data.
    output_string: string
        Name of numpy file that will be saved.
    """
    for k, v in data.iteritems():
        output = os.path.join(
            output_string,
            '%s_%s' % (model_name, k)
            )
        np.save(output, v)


def check_path(data_pointer, log, msg):
    """
    Check that the path exists.

    Parameters
    ----------
    data_pointer : string
        Directory to data.

    Returns
    -------
    bool
        False if path does not exist.
    data_pointer: string
        Directory to data if path does exist.
    """
    if not os.path.exists(data_pointer):
        log.debug(msg)
        return False
    else:
        return data_pointer


def ifloor(x):
    """
    Floor as an integer.

    Parameters
    ----------
    x : float

    Returns
    -------
    int
        Floor of x as an integer.
    """
    return np.floor(x).astype(np.int)


def iceil(x):
    """
    Ceiling as an integer.

    Parameters
    ----------
    x : float

    Returns
    -------
    int
        Ceil of x as an integer.
    """
    return np.ceil(x).astype(np.int)


def convert_to_tuple(v):
    """
    Convert v to a tuple.

    Parameters
    ----------
    v

    Returns
    -------
    v : tuple
    """
    if not isinstance(v, tuple):
        return tuple(v)
    else:
        return v
