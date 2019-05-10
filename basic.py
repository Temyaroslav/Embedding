
# Attempt to apply embedding using first minima of time-delayed mutual information
# and dimension with FNN below 10%.
# KNN is used to produce forecasts

from pathlib import PurePath, Path
import copy

import numpy as np
import pandas as pd

import embedding
import classifiers

pp = PurePath(Path.cwd()).parts
pdir = PurePath(*pp)

bid, ask = pd.read_csv(str(pdir) + '/data/eurusd-bid-1h.csv'), pd.read_csv(str(pdir) + '/data/eurusd-ask-1h.csv')
mids = ((bid.iloc[:, 1] + ask.iloc[:, 1]) / 2).dropna()

_embedding = embedding.Embedding(mids)
time_delayed_mi = _embedding.time_delayed_mutual_information()
# _embedding.plot_mutual_information(time_delayed_mi)

# First minima of time-delayed mutual information
time_delay = _embedding.locmin(time_delayed_mi)[0]
# _embedding.plot_delayed_series(tau=time_delay)

# Calculate FNN in the range of 10 dimensions. Takes some time to calculate!
# dim = np.arange(1, 10 + 1)
# f1, f2, f3 = _embedding.fnn(mids.values, dim=dim, tau=time_delay, window=10, metric='cityblock')
# _embedding.plot_fnn(dim, f1, f2, f3)

# judging from the plot above FNN goes beyond 10% in dim=4

m = 4

embedded = _embedding.embedding(tau=time_delay, m=m)
print('Number of embedding dimensions {0}'.format(len(embedded[0])))

embedding_featureset = []
embedding_labels = []

for i, vector in enumerate(embedded):
    if (i + 1) >= len(embedded):
        break

    embedding_featureset.append(vector)
    embedding_labels.append(embedded[i + 1])

assert len(embedding_featureset) == len(embedding_labels), "Did not get equal amount of predictions as points"

fraction = 8.0 / 10.0
train_set = (embedding_featureset[:int(fraction * len(embedding_featureset))],
                embedding_labels[:int(fraction * len(embedding_featureset))])
test_set = (embedding_featureset[int(fraction * len(embedding_featureset)):],
                embedding_labels[int(fraction * len(embedding_featureset)):])

X_train, y_train = train_set[0], train_set[1]
X_test, y_test = test_set[0], test_set[1]

knn = classifiers.KNN(X_train, y_train, k=5)

prediction = []; truth = []
train = copy.copy(X_train)
for i in range(len(X_train)):
    xx, yy = train[i], y_train[i]
    prediction_embedded = knn.classify(xx)

    prediction.append(prediction_embedded[0])
    truth.append(yy[0])

# Calculate the average error
training = [row[0] for row in X_train]
print("Error on training set: {0}".format(knn.error(prediction, truth, training)))

prediction = []; truth = []
test = copy.copy(X_test)
for i in range(len(X_test)):
    xx, yy = test[i], y_test[i]
    prediction_embedded = knn.classify(xx)

    prediction.append(prediction_embedded[0])
    truth.append(yy[0])

# Calculate the average error
testing = [row[0] for row in X_test]
print("Error on test set: {0}".format(knn.error(prediction, truth, testing)))


