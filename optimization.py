
# Here we will use "brute-force" method to find optimal time-delay and dimension
# which will simply include iteration through the domain of delays and dimensions we defined in basic.py

from pathlib import PurePath, Path

import numpy as np
import pandas as pd

import embedding


pp = PurePath(Path.cwd()).parts
pdir = PurePath(*pp)

files = ['eurusd_30m.csv',
         'eurusd_15m.csv',
         'eurusd_5m.csv']

time_delays = np.unique(np.logspace(0, 2, num=10, dtype='int'))
dims = np.array(range(2, 10 + 1))

_optimize = embedding.Optimize(time_delays, dims)

for file in files:
    data = pd.read_csv(str(pdir) + '/data/' + file)
    mids = ((data.bid_close + data.ask_close) / 2).dropna()

    data = mids.iloc[: int(0.8 * len(mids))]

    results = _optimize.brute_force(data)
    _optimize.save_brute_results(results, file)

