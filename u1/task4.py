"""Analysis of histograms and statistical deviations for the generators
Generate random numbers with all three generators. When you generate random
numbers with the two rejection methods, are the acceptance rates reasonable?

For all generators, perform a frequentist analysis to calculate histograms with
error bars. For the first generator (Cauchy distribution), also perform a Bayesian
analysis (see lecture notes) to calculate the histogram and error estimates.

In all cases, plot the histograms together with the error bars and the desired pdf.
Are the sizes of the ”error bars” reasonable? What size should you (roughly) ex-
pect? Do the sizes scale correctly with the number N of entries in the histogram?
Are the histograms compatible with the desired distributions, given the uncer-
tainties?

When do the differences between the frequentist and the Bayesian analysis be-
come visible ?
"""
import matplotlib.pyplot as plt
import numpy as np

from .base import RESULTS
from .distributions import DistributionType, cad, ecauchy, g, max_x, min_x

gen1 = ecauchy.inverse_transformation_rvs
gen2 = lambda: g.rejection_rvs(c=1 / g.normalization, envelope=ecauchy)
gen3 = lambda: g.rejection_rvs(c=1, envelope=cad)


def frequentist_expectation_std(occurences):
    """Given the raw occurences (heights) from the data, what are p_i and sigma_i?
    Based on the Frequentist formula (A.4 in lecture notes)"""
    return occurences / sum(occurences), np.sqrt(occurences) / sum(occurences)


def bayesian_expectation_std(occurences, bins):
    """Given the raw occurences (heights) from the data, what are p_i and sigma_i?
    Based on the Bayesian formula (A.14, A.15 in lecture notes)"""
    num = sum(occurences)
    expectation_values = (occurences + 1) / (num + bins + 1)
    expected_stds = np.sqrt(expectation_values * (1 - expectation_values) / (num + bins + 2))
    return expectation_values, expected_stds


def plot_frequentist_histo(axes, samples, dist: DistributionType, bins=80):
    """Using the frequentist approach, plots a detailed histogram."""
    heights, edges = np.histogram(samples, bins=bins, range=(min_x, max_x))
    bincenters = 0.5 * (edges[1:] + edges[:-1])

    bin_width = (max_x - min_x) / bins
    expectation_values, expected_stds = frequentist_expectation_std(heights)
    axes.bar(
        bincenters,
        expectation_values / bin_width,
        width=bin_width * 0.95,
        yerr=expected_stds / bin_width,
        label="histogram",
    )
    x = np.linspace(min_x, max_x, 100)
    axes.plot(x, dist.pdf(x), label=f"{dist.name} pdf", c="orange")
    axes.set_ylabel("$p(x)$")
    axes.legend()


def plot_bayes_histo(axes, samples, dist: DistributionType, bins=80):
    """Use the Bayesian approach"""
    heights, edges = np.histogram(samples, bins=bins, range=(min_x, max_x))
    bincenters = 0.5 * (edges[1:] + edges[:-1])

    bin_width = (max_x - min_x) / bins
    x = np.linspace(min_x, max_x, 100)
    expectation_values, expected_stds = bayesian_expectation_std(heights, bins)
    axes.bar(bincenters, expectation_values / bin_width, width=bin_width * 0.95, yerr=expected_stds, label="histogram")
    axes.plot(x, dist.pdf(x), label="pdf", c="orange")
    axes.set_ylabel("$p(x)$")
    axes.legend()


# pylint: disable=too-many-locals
def plot_error_dependency(gen, bins=30, index=15):
    """With different N, what uncertainties do we obtain?"""
    nums = np.logspace(2, 5, num=8)
    expectations_f, expectations_b = [], []
    uncertainties_f, uncertainties_b = [], []
    for num in nums:
        samples = [gen() for _ in range(int(num))]
        heights, _ = np.histogram(samples, bins=bins, range=(min_x, max_x))
        e_f, u_f = frequentist_expectation_std(heights)
        e_b, u_b = bayesian_expectation_std(heights, bins)
        expectations_f.append(e_f[index])
        uncertainties_f.append(u_f[index])
        expectations_b.append(e_b[index])
        uncertainties_b.append(u_b[index])

    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(2, 1, 1)
    axes.semilogx(nums, expectations_f, label="frequentist")
    axes.semilogx(nums, expectations_b, label="bayesian")
    axes.set_xlabel("$N$")
    axes.set_ylabel("$p_x(N)$")
    axes.set_title("Expectation values")
    axes.legend()
    axes: plt.Axes = fig.add_subplot(2, 1, 2)
    axes.semilogx(nums, uncertainties_f, label="frequentist")
    axes.semilogx(nums, uncertainties_b, label="bayesian")
    axes.set_xlabel("$N$")
    axes.set_ylabel(r"$\sigma_x(N)$")
    axes.set_title("Uncertainties")
    axes.legend()
    fig.savefig(RESULTS / "task4-error-dependency.png")


def main():
    """Run entire task4"""
    samples1 = [gen1() for _ in range(1000)]
    samples2 = [gen2() for _ in range(1000)]
    print(f"Rejections average for gen2: {np.mean(g.rejection_log)} ({len(g.rejection_log)} entries)")
    g.rejection_log.clear()
    samples3 = [gen3() for _ in range(1000)]
    print(f"Rejections average for gen3: {np.mean(g.rejection_log)} ({len(g.rejection_log)} entries)")
    g.rejection_log.clear()

    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(3, 1, 1)
    plot_frequentist_histo(axes, samples1, ecauchy)
    axes: plt.Axes = fig.add_subplot(3, 1, 2)
    plot_frequentist_histo(axes, samples2, g)
    axes: plt.Axes = fig.add_subplot(3, 1, 3)
    plot_frequentist_histo(axes, samples3, g)
    axes.set_xlabel("$x$")
    fig.savefig(RESULTS / "task4-frequentist.png")

    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(1, 1, 1)
    plot_bayes_histo(axes, samples1, ecauchy)
    axes.set_xlabel("$x$")
    fig.savefig(RESULTS / "task4-bayesian.png")

    plot_error_dependency(gen1)

    plt.show()
