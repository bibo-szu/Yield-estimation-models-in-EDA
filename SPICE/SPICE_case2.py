"""
    This is a fitted toy function. If you want to conduct circuit experiment,
    you should use a real SPICE simulator to replace this.
"""

import numpy as np
import torch.nn as nn
import joblib
import os
import pickle
import pandas as pd

def save_obj(obj,fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class SPICE_Case2(nn.Module):
    threshold = 5.31772498764667

    def __init__(self) -> None:
        super(SPICE_Case2, self).__init__()
        self.path = '.\SPICE\case2_lasso.pkl'
        self.model = joblib.load(self.path)
        self.feature_num = 18
        self.low = -4.16957
        self.up = 4.16957
        self.bound = [[self.low, self.up] for i in range(self.feature_num)]
        bounds = np.loadtxt('.\SPICE\model_2_bound.txt', dtype=np.float64)
        self.up_bounds = bounds[0,:]
        self.low_bounds = bounds[1,:]
        self.bound = np.vstack([self.low_bounds, self.up_bounds])
        self.case = 2

    def forward(self, x):
        return self.model.predict(x)[:,None]

    def get_initial_x(self, N):
        D = self.feature_num

        bounds = self.bound
        result = np.empty([N, D])
        temp = np.empty([N])
        d = 1.0 / N
        for i in range(D):
            for j in range(N):
                temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d, size=1)[0]
            np.random.shuffle(temp)
            for j in range(N):
                result[j, i] = temp[j]

        b = np.array(bounds)
        lower_bounds = b[0, :]
        upper_bounds = b[1, :]
        if np.any(lower_bounds > upper_bounds):
            print('Bounds Error')
            return None
        np.add(np.multiply(result,(upper_bounds - lower_bounds),out=result),
               lower_bounds,
               out=result)
        return result


    def indicator(self, y):
        return y > self.threshold

    def get_yield(self, num=1000000, std=1):
        x = np.random.multivariate_normal(mean=[0] * self.feature_num, cov=np.eye(self.feature_num)*std, size=num)
        fail_num = self.indicator(self(x)).sum()
        return fail_num / num

