import utils
from pathlib import PurePath, Path
pp = PurePath(Path.cwd()).parts
pdir = PurePath(*pp)

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.spatial import KDTree
from sklearn.model_selection import TimeSeriesSplit

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import classifiers
import utils


class Embedding:
    def __init__(self, data):
        self.data = data

    def _mutual_information(self, x, y, bins=64):
        """
        fn: calc mutual information between two random variables,
            I = S(x) + S(y) - S(x,y), between two
            random variables x and y, where S(x) is the Shannon entropy

        :param x: 1D array, first var
        :param y: 1D array, second var
        :param bins: int, number of bins for histogram
        :return: float, mutual information
        """
        p_x = np.histogram(x, bins)[0]
        p_y = np.histogram(y, bins)[0]
        p_xy = np.histogram2d(x, y, bins)[0].flatten()

        # Convert frequencies into probabilities.
        p_x = p_x[p_x > 0] / np.sum(p_x)
        p_y = p_y[p_y > 0] / np.sum(p_y)
        p_xy = p_xy[p_xy > 0] / np.sum(p_xy)

        # Calculate the corresponding Shannon entropies.
        h_x = np.sum(p_x * np.log2(p_x))
        h_y = np.sum(p_y * np.log2(p_y))
        h_xy = np.sum(p_xy * np.log2(p_xy))

        return h_xy - h_x - h_y

    def time_delayed_mutual_information(self, maxtau=1000, bins=64):
        """
        fn: calc mutual information between x_i and x_{i + t} (i.e., the
            time-delayed mutual information)

        :param x: 1D array of time series
        :param maxtau: int, time delay
        :param bins: int, number of bins for histogram
        :return: array with time-delayed mutual information up to maxtau
        """
        N = len(self.data)
        maxtau = min(N, maxtau)

        ii = np.empty(maxtau)

        for tau in range(1, maxtau):
            ii[tau] = self._mutual_information(self.data[:-tau], self.data[tau:], bins)

        return ii

    def locmin(self, x, ):
        """
        fn: All local minimas of an array
        :param x: 1D array
        :return: array of local minimas
        """
        return (np.diff(np.sign(np.diff(x))) > 0).nonzero()[0] + 1

    def fnn(self, x, dim=[1], tau=1, R=10.0, A=2.0, metric='euclidean', window=10,
            maxnum=None, parallel=True):
        """Compute the fraction of false nearest neighbors.

        Implements the false nearest neighbors (FNN) method described by
        Kennel et al. (1992) to calculate the minimum embedding dimension
        required to embed a scalar time series.

        Parameters
        ----------
        x : array
            1-D real input array containing the time series.
        dim : int array (default = [1])
            Embedding dimensions for which the fraction of false nearest
            neighbors should be computed.
        tau : int, optional (default = 1)
            Time delay.
        R : float, optional (default = 10.0)
            Tolerance parameter for FNN Test I.
        A : float, optional (default = 2.0)
            Tolerance parameter for FNN Test II.
        metric : string, optional (default = 'euclidean')
            Metric to use for distance computation.  Must be one of
            "cityblock" (aka the Manhattan metric), "chebyshev" (aka the
            maximum norm metric), or "euclidean".  Also see Notes.
        window : int, optional (default = 10)
            Minimum temporal separation (Theiler window) that should exist
            between near neighbors.
        maxnum : int, optional (default = None (optimum))
            Maximum number of near neighbors that should be found for each
            point.  In rare cases, when there are no neighbors that are at a
            nonzero distance, this will have to be increased (i.e., beyond
            2 * window + 3).
        parallel : bool, optional (default = True)
            Calculate the fraction of false nearest neighbors for each d
            in parallel.

        Returns
        -------
        f1 : array
            Fraction of neighbors classified as false by Test I.
        f2 : array
            Fraction of neighbors classified as false by Test II.
        f3 : array
            Fraction of neighbors classified as false by either Test I
            or Test II.

        """
        if parallel:
            processes = None
        else:
            processes = 1

        return utils.parallel_map(self._fnn, dim, (x,), {
            'tau': tau,
            'R': R,
            'A': A,
            'metric': metric,
            'window': window,
            'maxnum': maxnum
        }, processes).T

    def _fnn(self, d, x, tau=1, R=10.0, A=2.0, metric='euclidean', window=10,
             maxnum=None):
        """ fn: Return fraction of false nearest neighbors for a single d.

            Returns the fraction of false nearest neighbors for a single d.
            This function is meant to be called from the main fnn() function.

        """

        # We need to reduce the number of points in dimension d by tau
        # so that after reconstruction, there'll be equal number of points
        # at both dimension d as well as dimension d + 1.
        y1 = self._reconstruct(x[:-tau], d, tau)
        y2 = self._reconstruct(x, d + 1, tau)

        # Find near neighbors in dimension d.
        index, dist = self._neighbors(y1, metric=metric, window=window,
                                      maxnum=maxnum)

        # Find all potential false neighbors using Kennel et al.'s tests.
        f1 = np.abs(y2[:, -1] - y2[index, -1]) / dist > R
        f2 = self._dist(y2, y2[index], metric=metric) / np.std(x) > A
        f3 = f1 | f2

        return np.mean(f1), np.mean(f2), np.mean(f3)

    def _dist(self, x, y, metric='chebyshev'):
        """
        fn: Compute the distance between all sequential pairs of points.

            Computes the distance between all sequential pairs of points from
            two arrays using scipy.spatial.distance.

        :param x: 1D input array
        :param y: 1D input array
        :param metric: string, optional (default = 'chebyshev')
            Metric to use while computing distances.
        :return: array with distances
        """
        func = getattr(distance, metric)
        return np.asarray([func(i, j) for i, j in zip(x, y)])

    def _reconstruct(self, x, dim=1, tau=1):
        """
        fn: Construct time-delayed vectors from a time series.

        :param x: 1D time-series
        :param dim: int, optional (default = 1)
            Embedding dimension.
        :param tau: int, optional (default = 1)
            Time delay
        :return: array with time-delayed vectors
        """
        m = len(x) - (dim - 1) * tau
        if m <= 0:
            raise ValueError('Length of the time series is <= (dim - 1) * tau.')

        return np.asarray([x[i:i + (dim - 1) * tau + 1:tau] for i in range(m)])

    def _neighbors(self, y, metric='chebyshev', window=0, maxnum=None):
        """
        fn: Find nearest neighbors of all points in the given array.

            Finds the nearest neighbors of all points in the given array using
            SciPy's KDTree search.

        :param y: ndarray
            N-dimensional array containing time-delayed vectors.
        :param metric: str, optional (default = 'chebyshev')
            Metric to use for distance computation.  Must be one of
            "cityblock" (aka the Manhattan metric), "chebyshev" (aka the
            maximum norm metric), or "euclidean".
        :param window: int, optional (default = 0)
            Minimum temporal separation (Theiler window) that should exist
            between near neighbors.  This is crucial while computing
            Lyapunov exponents and the correlation dimension.
        :param maxnum: nt, optional (default = None (optimum))
            Maximum number of near neighbors that should be found for each
            point.  In rare cases, when there are no neighbors that are at a
            nonzero distance, this will have to be increased (i.e., beyond
            2 * window + 3).
        :return:
            index: array
                Array containing indices of near neighbors.
            dist: array
                Array containing near neighbor distances.
        """
        if metric == 'cityblock':
            p = 1
        elif metric == 'euclidean':
            p = 2
        elif metric == 'chebyshev':
            p = np.inf
        else:
            raise ValueError('Unknown metric.  Should be one of "cityblock", '
                             '"euclidean", or "chebyshev".')

        tree = KDTree(y)
        n = len(y)

        if not maxnum:
            maxnum = (window + 1) + 1 + (window + 1)
        else:
            maxnum = max(1, maxnum)

        if maxnum >= n:
            raise ValueError('maxnum is bigger than array length.')

        dists = np.empty(n)
        indices = np.empty(n, dtype=int)

        for i, x in enumerate(y):
            for k in range(2, maxnum + 2):
                dist, index = tree.query(x, k=k, p=p)
                valid = (np.abs(index - i) > window) & (dist > 0)

                if np.count_nonzero(valid):
                    dists[i] = dist[valid][0]
                    indices[i] = index[valid][0]
                    break

                if k == (maxnum + 1):
                    raise Exception('Could not find any near neighbor with a '
                                    'nonzero distance.  Try increasing the '
                                    'value of maxnum.')

        return np.squeeze(indices), np.squeeze(dists)

    def embedding(self, tau, m):
        """

        fn: Given time-delay and number of dimension returns embedded data

        :param data: 1D time-series
        :param tau: int, time-delay
        :param m: int, dimension
        :return: md array with embedded data
        """
        points = []

        for i, row in enumerate(self.data.values):
            point = []
            for j in range(m):
                if i + j * tau < len(self.data.values):
                    point.append(self.data.values[i + j * tau])

            if len(point) == m:
                points.append(point)

        return points

    def plot_mutual_information(self, tdmi):
        plt.title('Delayed mutual information of ccy pair')
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$I(\tau)$')
        plt.plot(tdmi)
        plt.show()

    def plot_delayed_series(self, tau):
        plt.title(r'Time delay = %d' % tau)
        plt.xlabel(r'$x(t)$')
        plt.ylabel(r'$x(t + \tau)$')
        plt.plot(self.data[:-tau], self.data[tau:])
        plt.show()

    def plot_fnn(self, dim, f1, f2, f3):
        plt.title(r'FNN for Ccy pair')
        plt.xlabel(r'Embedding dimension $d$')
        plt.ylabel(r'FNN (%)')
        plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')
        plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')
        plt.plot(dim, 100 * f3, 'rs-', label=r'Test I + II')
        plt.axhline(y=10, color='k', linestyle='dashed')
        plt.legend()

        plt.show()


class Optimize:
    def __init__(self, time_delays, dims):
        self.time_delays = time_delays
        self.dims = dims
        self.tscv = TimeSeriesSplit(n_splits=3)

    def brute_force(self, x, parallel=True):
        if parallel:
            processes = None
        else:
            processes = 1

        results = []
        for tau in self.time_delays:
            parallel_results = utils.parallel_map(self._brute_force, self.dims, (x,), {'tau': tau}, processes)
            results += np.array(parallel_results).T.tolist()

        return results

    def _brute_force(self, d, x, tau=1):
        features = [];
        labels = []
        _embedding = Embedding(x)
        embedded = _embedding.embedding(tau=tau, m=d)

        for i, vector in enumerate(embedded):
            if (i + 1) >= len(embedded):
                break
            features.append(vector)
            labels.append(embedded[i + 1])

        features = np.array(features)
        labels = np.array(labels)

        error_train, error_validation = [], []

        for train_index, validation_index in self.tscv.split(features):
            X_train, X_validation = features[train_index], features[validation_index]
            y_train, y_validation = labels[train_index], labels[validation_index]

            ppmd = classifiers.PPMD(X_train, y_train)

            prediction = []
            truth = []
            for i in range(len(X_train)):
                pred = ppmd.classify(X_train[i])
                prediction.append(pred)
                truth.append(y_train[i])

            error_train.append(ppmd.error(prediction, truth))

            ppmd = classifiers.PPMD(X_validation, y_validation)

            prediction = []
            truth = []
            for i in range(len(X_validation)):
                pred = ppmd.classify(X_validation[i])
                prediction.append(pred)
                truth.append(y_validation[i])

            error_validation.append(ppmd.error(prediction, truth))

        error_train = np.average(np.array(error_train))
        error_validation = np.average(np.array(error_validation))

        temp_dict = {'dimension': d,
                     'delay': tau,
                     'MASE_train': error_train,
                     'MASE_validation': error_validation}

        print('For dimension = {0} and delay = {1} Errors on TRAIN = {2}, VALIDATE = {3}'
              .format(d, tau, error_train, error_validation))

        return temp_dict

    def save_brute_results(self, results, name):
        # assert isinstance(results, pd.DataFrame), 'Results should be pd.DataFrame() type'
        results = pd.DataFrame(results)
        results.to_csv(str(pdir) + "/signals/MASE_" + name, index=False)
        return

