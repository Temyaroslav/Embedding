import numpy as np
import pandas as pd


from multiprocessing import Pool

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def parallel_map(func, values, args=tuple(), kwargs=dict(),
                 processes=None):
    """
    fn: Use Pool.apply_async() to get a parallel map().

    Uses Pool.apply_async() to provide a parallel version of map().
    Unlike Pool's map() which does not let you accept arguments and/or
    keyword arguments, this one does.

    :param func: function
        This function will be applied on every element of values in
        parallel.
    :param values: array
        Input array
    :param args: tuple, optional (default: ())
        Additional arguments for func.
    :param kwargs: dictionary, optional (default: {})
        Additional keyword arguments for func.
    :param processes: int, optional (default: None)
        Number of processes to run in parallel.  By default, the output
        of cpu_count() is used.
    :return: array
        Output after applying func on each element in values.
    """

    # True single core processing, in order to allow the func to be executed in
    # a Pool in a calling script.
    if processes == 1:
        return np.asarray([func(value, *args, **kwargs) for value in values])

    pool = Pool(processes=processes)
    results = [pool.apply_async(func, (value,) + args, kwargs)
               for value in values]

    pool.close()
    pool.join()

    return np.asarray([result.get() for result in results])
