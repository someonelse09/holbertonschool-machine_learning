#!/usr/bin/env python3
"""This module contains the class that performs
Bayesian optimization on a noiseless 1D Gaussian process"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """
    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Args:
            f is the black-box function to be optimized
            X_init is a numpy.ndarray of shape (t, 1) representing
             the inputs already sampled with the black-box function
            Y_init is a numpy.ndarray of shape (t, 1) representing
             the outputs of the black-box function for each input in X_init
            t is the number of initial samples
            bounds is a tuple of (min, max) representing the
             bounds of the space in which to look for the optimal point
            ac_samples is the number of samples that
             should be analyzed during acquisition
            l is the length parameter for the kernel
            sigma_f is the standard deviation given
             to the output of the black-box function
            xsi is the exploration-exploitation factor for acquisition
            minimize is a bool determining whether optimization
             should be performed for minimization
             (True) or maximization (False)
        Sets the following public instance attributes:
            f: the black-box function
            gp: an instance of the class GaussianProcess
            X_s: a numpy.ndarray of shape (ac_samples, 1) containing
             all acquisition sample points, evenly spaced between min and max
            xsi: the exploration-exploitation factor
            minimize: a bool for minimization versus maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """ Calculates the next best sample location
        Args:
            Uses the Expected Improvement acquisition function
        Returns: X_next, EI
            X_next is a numpy.ndarray of shape (1,)
             representing the next best sample point
            EI is a numpy.ndarray of shape (ac_samples,) containing
             the expected improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)
        # Find the current best observed value
        if self.minimize:
            Y_best = np.min(self.gp.Y)
            # For minimization, improvement is Y_best - mu
            improvement = Y_best - mu
        else:
            Y_best = np.max(self.gp.Y)
            # For maximization, improvement is mu - Y_best
            improvement = mu - Y_best
        # Adding exploration factor xsi to the improvement
        improvement_with_xsi = improvement - self.xsi
        # Calculate Expected Improvement (EI)
        with np.errstate(divide='warn', invalid='warn'):
            z = np.zeros_like(sigma)
            non_zero = sigma > 0
            z[non_zero] = improvement_with_xsi[non_zero] / sigma[non_zero]
            # EI formula: EI = improvement * CDF(Z) + sigma * PDF(Z)
            EI = np.zeros_like(sigma)
            EI[non_zero] = \
                           (improvement_with_xsi[non_zero] * norm.cdf(z[non_zero])) +\
                           sigma[non_zero] * norm.pdf(z[non_zero])
            EI[~non_zero] = np.maximum(improvement_with_xsi[~non_zero], 0)
        # Find the point with maximum EI
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Args:
            iterations is the maximum number of iterations to perform
            If the next proposed point is one that has already been sampled,
             optimization should be stopped early
        Returns:
            X_opt, Y_opt
            X_opt is a numpy.ndarray of shape (1,)
             representing the optimal point
            Y_opt is a numpy.ndarray of shape (1,)
             representing the optimal function value
        """
        for i in range(iterations):
            # Get next point to sample using acquisition function
            X_next, _ = self.acquisition()
            # Check if this point has already been sampled
            # If so, stop early (convergence)
            if np.any(np.isclose(self.gp.X, X_next)):
                break
            # Evaluate the black-box function at the new point
            Y_next = self.f(X_next)
            # Update the Gaussian Process with the new observation
            self.gp.update(X_next, Y_next)
        # Find the optimal point from all sampled points
        if self.minimize:
            idx_opt = np.argmin(self.gp.Y)
        else:
            idx_opt = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx_opt]
        Y_opt = self.gp.Y[idx_opt]

        return X_opt, Y_opt
