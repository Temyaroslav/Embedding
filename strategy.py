from pathlib import PurePath, Path
pp = PurePath(Path.cwd()).parts
pdir = PurePath(*pp)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.svm import NuSVR

from embedding import Embedding

data = pd.read_csv(str(pdir) + '/data/eurusd_30m.csv')
data['mid_close'] = ((data.bid_close + data.ask_close)/2)
bids = data.iloc[int(0.8 * len(data)):].bid_open
asks = data.iloc[int(0.8 * len(data)):].ask_open
mids = data.mid_close

_embedding = Embedding(mids)
embedded = _embedding.embedding(1, 3)
embedded = np.array(embedded)

features = []; labels = []
for i, vector in enumerate(embedded):
    if (i + 1) >= len(embedded):
        break
    features.append(vector)
    labels.append(embedded[i + 1])

fraction = 0.8
X_train, y_train = features[:int(fraction * len(features))], labels[:int(fraction * len(labels))]
X_test, y_test = features[int(fraction * len(features)):], labels[int(fraction * len(labels)):]

print('TRAIN: X - {0}; y - {1}'.format(len(X_train), len(y_train)))
print('TEST: X - {0}; y - {1}'.format(len(X_test), len(y_test)))

# need to convert to np.array() to extract last values
y_train = np.array(y_train)
y_test = np.array(y_test)

y_train = y_train[:, y_train.shape[1]-1]
y_test = y_test[:, y_test.shape[1]-1]

# make candles
s = data.index[data['mid_close'] == y_test[0]].tolist()[1]
px_test = {'bid': data.iloc[s:,1].values,
           'ask': data.iloc[s:,5].values,
           'mid': data.iloc[s:,9].values}

clf = NuSVR()
fitted_clf = clf.fit(X_train, y_train)


def plot_strategy(strategy, default):
    '''
    fn: compare 2 strategies

    Params:
    -------
    strategy: list, accumulated returns from predicting strateg
    default: list, accumulated returns from buy & hold
    '''
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    x = range(len(strategy))
    ax.plot(x, strategy, label='strategy')
    ax.plot(x, default, c='r', label='buy & hold')
    plt.legend()
    plt.show()


order_book = {'side': None, 'lvl': None}
p = []; s = 0
for i in range(len(X_test)):
    pred = fitted_clf.predict(X_test[i].reshape(1, -1))[0]
    pred_sign = np.sign(pred - X_test[i][len(X_test[i])-1])
    if pred_sign > 0:
        if order_book['side'] == 's' or order_book['side']is None:
            order_book['side'] = 'b'
            if order_book['lvl'] is not None:
                s += order_book['lvl'] - px_test['ask'][i]
            order_book['lvl'] = px_test['ask'][i]
    elif pred_sign < 0:
        if order_book['side'] == 'b' or order_book['side'] is None:
            order_book['side'] = 's'
            if order_book['lvl'] is not None:
                s += px_test['bid'][i] - order_book['lvl']
            order_book['lvl'] = px_test['bid'][i]
    px = px_test['mid'][i]
    p.append(s)

