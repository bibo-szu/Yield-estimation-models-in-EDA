"""
    normal distribution
"""

from math import pi
from scipy.special import logsumexp
import mpmath as mp
import numpy as np
from scipy.stats import multivariate_normal

class norm_dist():
    def __init__(self, mu, var):
        """
        :param mu: d
        :param var: k x d
        """
        self.mu = mu
        self.var = var

    def sample(self, n):
        return np.random.multivariate_normal(mean=self.mu, cov=self.var, size=n)

    def log_pdf(self, x):
        return multivariate_normal(mean=self.mu, cov=self.var).logpdf(x)

    def log_cdf(self, x):
        return multivariate_normal(mean=self.mu, cov=self.var).logcdf(x)

    def pdf(self, x, mp_format=False):
        log_pdf = self.log_pdf(x)
        mp_exp_broad = np.frompyfunc(mp.exp, 1, 1)
        prob = mp_exp_broad(log_pdf)
        if mp_format==False:
            prob = prob.astype(float)
        return prob

if __name__ == "__main__":

    pass