from pathlib import PurePath, Path
pp = PurePath(Path.cwd()).parts
pdir = PurePath(*pp)

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import TimeSeriesSplit

from classifiers import PPMD
from classifiers import GridSearchTS


def _grid_search(k):
    temp_dict = []
    for metric in grid_params['metric']:
        error_train, error_validation = [], []
        sign_train, sign_validation = [], []
        for train_index, validation_index in tscv.split(gs.features):
            X_train, X_validation = gs.features[train_index], gs.features[validation_index]
            y_train, y_validation = gs.labels[train_index], gs.labels[validation_index]

            ppmd = PPMD(X_train, y_train, k=k, metric=metric)

            prediction = [];
            truth = [];
            s = 0
            print('Calculating for TRAIN:\n')
            for i in tqdm(range(len(X_train))):
                pred = ppmd.classify(X_train[i])
                prediction.append(pred)
                truth.append(y_train[i])

                if np.sign(np.median(pred) - np.median(X_train[i])) == np.sign(
                        np.median(y_train[i]) - np.median(X_train[i])):
                    s += 1

            sign_train.append(s / len(prediction))
            error_train.append(ppmd.error(prediction, truth))

            print('Calculating for VALIDATION:\n')
            prediction = [];
            truth = [];
            s = 0
            for i in tqdm(range(len(X_validation))):
                pred = ppmd.classify(X_validation[i])
                prediction.append(pred)
                truth.append(y_validation[i])

                if np.sign(np.median(pred) - np.median(X_validation[i])) == np.sign(
                        np.median(y_validation[i]) - np.median(X_validation[i])):
                    s += 1

            sign_validation.append(s / len(prediction))
            error_validation.append(ppmd.error(prediction, truth))

        error_train = np.average(np.array(error_train))
        error_validation = np.average(np.array(error_validation))
        sign_train = np.average(np.array(sign_train))
        sign_validation = np.average(np.array(sign_validation))

        d = {'neighbors': k,
             'metric': metric,
             'MASE_train': error_train,
             'MASE_validation': error_validation,
             'SIGN_train': sign_train,
             'SIGN_validation': sign_validation}

        temp_dict.append(d)

    return temp_dict


if __name__ == '__main__':
    data = pd.read_csv(str(pdir) + '/data/eurusd_5m.csv')
    mids = ((data.bid_close + data.ask_close) / 2).dropna()
    data = mids.iloc[: int(0.8 * len(mids))]

    grid_params = {'n_neighbors': list(np.linspace(2, 10, 9, dtype=int)),
                    # 'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    tscv = TimeSeriesSplit(n_splits=3)

    # Using dim and tau from brute_force optimization
    gs = GridSearchTS(data, dim=4, tau=1, grid_params=grid_params)
    res = gs.grid_search(_grid_search)

    gs.save_grid_search(res, 'KNN.csv')

