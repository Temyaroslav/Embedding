from pathlib import PurePath, Path
pp = PurePath(Path.cwd()).parts
pdir = PurePath(*pp)

import multiprocessing

import numpy as np
import pandas as pd

from sklearn.neighbors import BallTree

import embedding


class KNN:
    def __init__(self, features, labels, k=5):
        """
        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        self._kdtree = BallTree(features)
        self._y = labels
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median of the majority labels (as implemented
        in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """
        assert len(item_indices) == self._k, "Did not get k inputs"

        # Get the labels of the k nearest neighbors
        knn_labels = []
        for item_index in item_indices:
            knn_labels.append(self._y[item_index])

        # Return the average of the next label
        average_vector = []
        for i in range(len(knn_labels[0])):
            average_vector.append(np.average([row[i] for row in knn_labels]))

        return average_vector

    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """

        dist, ind = self._kdtree.query(np.array(example).reshape(1, -1), k=self._k)

        return self.majority(ind[0])

    def error(self, prediction, truth, training):
        """
        Compute the accuracy of our forecast
        """

        assert len(prediction) == len(truth), "Number of predictions much equal the number of real data points."

        error = np.abs(np.array(prediction) - np.array(truth))
        random_walk = np.abs(np.diff(training, axis = 0)).sum()
        return np.array(error/((len(prediction)/(len(training) - 1.0))*random_walk)).sum()


class PPMD:
    def __init__(self, features, labels, k=5):
        self.features = features
        self._kdtree = BallTree(features)
        self._y = labels
        self._k = k

    def majority(self, label_indices):
        assert len(label_indices) == self._k, "Did not get k inputs"

        # Get the labels of the k nearest neighbors
        knn_labels = []
        for label_index in label_indices:
            knn_labels.append(self._y[label_index])

        # Return the median of the next label
        median_vector = []
        for i in range(len(knn_labels[0])):
            median_vector.append(np.median([row[i] for row in knn_labels]))

        return median_vector

    def classify(self, feature_set):
        # ind = self._kdtree.query_radius(feature_set.reshape(1, -1), r=1)
        dist, ind = self._kdtree.query(feature_set.reshape(1, -1), k=self._k)

        return self.majority(ind[0])

    def error(self, prediction, truth, features):

        assert len(prediction) == len(truth), "Number of predictions must equal the number of real data points."

        error = np.abs(np.array(prediction) - np.array(truth))
        random_walk = np.abs(np.diff(features, axis=0)).sum()

        return np.array(error/((len(prediction)/(len(features) - 1.0))*random_walk)).sum()


class GridSearchTS:
    def __init__(self, data, dim, tau, grid_params):
        self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

        _embedding = embedding.Embedding(data)
        self.embedded = _embedding.embedding(tau=tau, m=dim)

        # transform to features and labels
        self.features = [];
        self.labels = []
        for i, vector in enumerate(self.embedded):
            if (i + 1) >= len(self.embedded):
                break
            self.features.append(vector)
            self.labels.append(self.embedded[i + 1])

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)

        assert isinstance(grid_params, dict), 'grid_params must be dict'
        self.grid_params = grid_params

    def grid_search(self, func):

        #         parallel_results = parallel_map(self._grid_search, (self.grid_params['n_neighbors'],), processes)
        #         results += np.array(parallel_results).T.tolist()

        results = self.pool.map(func, self.grid_params['n_neighbors'])
        self.pool.close()

        return results

    def save_grid_search(self, results, name):
        # assert isinstance(results, pd.DataFrame), 'Results should be pd.DataFrame() type'
        results = pd.DataFrame(results)
        results.to_csv(str(pdir) + "/signals/GridSearch_" + name, index=False)
        return


