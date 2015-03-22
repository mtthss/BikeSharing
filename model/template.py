from __future__ import division, print_function

import numpy as np


class IntegratedRegressor():
    regs = []

    def __init__(self, reg, reg_args, predict_log=True):
        self.reg = reg
        self.reg_args = reg_args
        self.predict_log = predict_log

    def fit(self, X, y):
        self.regs = []
        for target in y.columns:
            tmp = self.reg(**self.reg_args)
            if self.predict_log:
                tmp.fit(X, np.log1p(y[target]))
            else:
                tmp.fit(X, y[target])
            self.regs.append(tmp)

    def predict(self, X):
        pred = np.zeros((X.shape[0],))
        for reg in self.regs:
            if self.predict_log:
                pred += np.expm1(reg.predict(X))
            else:
                pred += reg.predict(X)
        return np.intp(pred.round())


class DayNightRegressor():
    def __init__(self, reg, reg_args):
        self.night_reg = reg(**reg_args)
        self.day_reg = reg(**reg_args)

    def fit(self, X, y):
        self.night_reg.fit(X[X['night'] == 1], y[X['night'] == 1])
        self.day_reg.fit(X[X['night'] == 0], y[X['night'] == 0])

    def predict(self, X):
        pred = []
        pred = np.append(pred, self.night_reg.predict(X[X['night'] == 1]))
        pred = np.append(pred, self.night_reg.predict(X[X['night'] == 0]))
        idx = X[X['night'] == 1].index.tolist() + X[X['night'] == 0].index.tolist()
        return np.intp(x for (_, x) in sorted(zip(idx, pred)))
