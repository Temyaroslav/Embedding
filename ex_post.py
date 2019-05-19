from pathlib import PurePath, Path
pp = PurePath(Path.cwd()).parts
pdir = PurePath(*pp)

import numpy as np
import pandas as pd

from embedding import Embedding


def dimension_fnn(x):
    _embedding = Embedding(x)

    dim = np.arange(1, 20 + 1)
    f1, f2, f3 = _embedding.fnn(x, tau=14, dim=dim, window=10, metric='cityblock')
    _embedding.plot_fnn(dim, f1, f2, f3)


def dimension_afn(x):
    _embedding = Embedding(x)

    dim = np.arange(1, 20 + 2)
    E, Es = _embedding.afn(x, tau=138, dim=dim, window=45, metric='chebyshev')
    E1, E2 = E[1:] / E[:-1], Es[1:] / Es[:-1]
    _embedding.plot_afn(dim, E1, E2)


if __name__ == '__main__':
    data = pd.read_csv(str(pdir) + '/data/eurusd_30m.csv')
    mids = ((data.bid_close + data.ask_close) / 2).dropna()
    data = mids.iloc[: int(0.8 * len(mids))]  # excluding hold-out

    test = mids.iloc[int(0.8 * len(mids)):]
    validate = mids.iloc[int(0.6 * len(mids)):int(0.8 * len(mids))]
    train = mids.iloc[:int(0.6 * len(mids))]

    dimension_afn(train.values)
