"""Histograms and their fluctuations:
In this first simple problem we will examine
histograms themselves. The description is detailed in order to make the task
straightforward. We use uniformly distributed random numbers for simplicity.
Generate them with a library routine.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from .base import RESULTS


def task_a(number_of_samples=1000, bins=50):
    """Generate N uniformly distributed random numbers (rand(N,1) in Matlab).
    Plot a histogram, normalized as a pdf. This means that for each bar, the
    count of each bin has to be normalized by N · b, where b is the bin width.
    (In Matlab, you can use histogram with appropriate parameters.) Vary N
    and look at the fluctuations of the bar heights. Do the fluctuations have
    about the expected size? You will have to zoom into the plot.
    """
    uniformly_distributed = np.random.rand(number_of_samples)
    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(1, 1, 1)
    heights, _, _ = axes.hist(uniformly_distributed, bins)
    mean, std = np.mean(heights), np.std(heights)  # type: ignore
    expected_std = np.sqrt(mean)
    axes.axhline(mean, linestyle="--", c="r", label=r"mean $\langle N_i \rangle$")
    axes.axhline(mean + std, linestyle="--", c="k", label=r"observed std-dev $\sigma_{N_i}$")
    axes.axhline(mean - std, linestyle="--", c="k")
    axes.axhline(mean + expected_std, linestyle="--", c="g", label=r"expected std-dev $\sqrt{\langle N_i \rangle}$")
    axes.axhline(mean - expected_std, linestyle="--", c="g")
    axes.set_xlabel("$x$")
    axes.set_ylabel("occurences")
    axes.legend(loc="lower left")
    fig.savefig(RESULTS / f"task1a-{number_of_samples}.png")


def task_b(number_of_supersamples=1000, superbins=50, number_of_samples=10000, bins=10):
    """We examine the fluctuations of the height of an individual bar. Generate
    L = 1000 samples of N random numbers each. For each sample, calcu-
    late (but do not plot) a histogram, normalized for a probability density
    function. Use just the height of the bar corresponding to random numbers
    0.5 < x < 0.6 (i.e. the sixth of 10 bars). What is the expectation value of this
    height ? Collect the L heights and plot a histogram of these values. How
    should the width of this histogram depend on N ? Verify (approximately)
    the expected behavior with a few examples, e.g. with N = 10m , m=3 and 5
    (m between 2 and 6 should work well). In Matlab, the commands histcounts
    and the very similar histogram make this easy.
    """
    height_5 = []
    for _ in range(number_of_supersamples):
        uniformly_distributed = np.random.rand(number_of_samples)
        heights, _ = np.histogram(uniformly_distributed, bins)
        height_5.append(heights[5])
    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(1, 1, 1)
    axes.hist(height_5, bins=superbins, density=True, label="Occurences with height[5]")
    loc, scale = scipy.stats.norm.fit(height_5)
    x = np.linspace(min(height_5), max(height_5))
    dist = scipy.stats.norm(loc, scale)
    axes.plot(x, dist.pdf(x), label="Normal distribution fit")
    axes.set_xlabel("height[5]")
    axes.set_ylabel("$p(x)$")
    axes.legend()
    fig.savefig(RESULTS / f"task1b-{number_of_samples}.png")


def task_c(number_of_samples=1000, bins=50):
    """Now we add error bars to the histograms of part (a). Again generate a hi-
    stogram of N uniform random numbers, normalized as a pdf. Estimate the
    size of the error of a given count Ni from the frequentist expression (see lec-
    ture notes). Draw the histogram with error bars. (In Matlab, error bars need
    to be added with the routine errorbar, which needs the x-coordinates of the
    bin-centers, the heights of the bars, and the (normalized) size of the errors
    as input. These numbers can be obtained like h=histogram(...); w=h.BinWidth
    etc.) Vary N and verify, using a few examples, that the size of the error-bars
    correspond to the fluctuations in the histograms.
    """
    uniformly_distributed = np.random.rand(number_of_samples)
    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(1, 1, 1)
    heights, edges = np.histogram(uniformly_distributed, bins)
    bincenters = 0.5 * (edges[1:] + edges[:-1])
    expected_stds = np.sqrt(heights)
    axes.bar(bincenters, heights, width=1 / bins, yerr=expected_stds, label="histogram")
    axes.set_xlabel("$x$")
    axes.set_ylabel("occurences")
    axes.legend(loc="lower left")
    fig.savefig(RESULTS / f"task1c-{number_of_samples}.png")


def main():
    """Run entire task1"""
    task_a(number_of_samples=100)
    task_a(number_of_samples=1000)
    task_a(number_of_samples=10000)
    task_b(number_of_samples=100)
    task_b(number_of_samples=1000)
    task_b(number_of_samples=10000)
    task_c(number_of_samples=100)
    task_c(number_of_samples=1000)
    task_c(number_of_samples=10000)
    plt.show()
