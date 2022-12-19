"""Inverse transformation, Cauchy distribution:
Write a program which generates random numbers according to the Cauchy
distribution
            1
h(x) = -----------
        π (1 + x²)
by using the inverse transformation method. You first need to find the corre-
sponding cumulative distribution function. You can use a library function for
generating uniform random numbers.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from .base import RESULTS
from .distributions import ecauchy, max_x, min_x


def main():
    """Run entire task2"""
    samples = 1000
    c = (max_x - min_x) / np.pi
    opts = dict(bins=50, range=(min_x, max_x))
    envelope = scipy.stats.uniform(min_x, max_x - min_x)

    fig = plt.figure()

    axes: plt.Axes = fig.add_subplot(3, 1, 1)
    axes.hist([ecauchy.inverse_transformation_rvs() for _ in range(samples)], label="Inverse-Transform-Cauchy", **opts)
    axes.legend()

    axes: plt.Axes = fig.add_subplot(3, 1, 2)
    axes.hist([ecauchy.rejection_rvs(c, envelope) for _ in range(samples)], label="Rejection-Sampling-Cauchy", **opts)
    axes.legend()

    axes: plt.Axes = fig.add_subplot(3, 1, 3)
    axes.hist([ecauchy.rvs() for _ in range(samples)], label="Scipy-Cauchy", **opts)
    axes.set_xlabel("$x$")
    axes.legend()

    fig.savefig(RESULTS / "task2.png")
    plt.show()
