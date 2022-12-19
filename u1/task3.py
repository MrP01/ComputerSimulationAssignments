"""Write a program that generates random numbers from the normalized probabi-
lity distribution
        1 1 sin2 (x)
g(x) =  ------------
        z π (1 + x2)
with z a normalizing constant (here z = (1 - e^(-2)) / 2).
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from .base import RESULTS
from .distributions import DistributionType, cad, g, max_x, min_x


def plot_g(c, envelope: DistributionType):
    """Plots histogram and PDF of g with a given envelope & c"""
    assert all(g.pdf(x) <= c * envelope.pdf(x) for x in np.linspace(min_x, max_x))
    g_samples = [g.rejection_rvs(c, envelope) for _ in range(10000)]
    x = np.linspace(min_x, max_x, 200)
    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(2, 1, 1)
    axes.hist(g_samples, 80, range=(min_x, max_x), label="RVS of $g(x)$")
    axes.set_xlabel("$x$")
    axes.set_ylabel("$p(x)$")
    axes.legend()
    axes: plt.Axes = fig.add_subplot(2, 1, 2)
    axes.plot(x, c * envelope.pdf(x), label=f"$h(x)$ ({envelope.name})")
    axes.plot(x, g.pdf(x), label="g(x)")
    axes.set_xlabel("$x$")
    axes.set_ylabel("$pdf(x)$")
    axes.set_ylim((0, 1))
    axes.legend()
    return fig


def verify_cad():
    """Samples numbers from the cad distribution and plots them next to the pdf"""
    cad_samples = [cad.rvs() for _ in range(1000)]
    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(1, 1, 1)
    x = np.linspace(min_x, max_x, 100)
    axes.hist(cad_samples, 80, range=(min_x, max_x), density=True, label="CAD histogram")
    axes.plot(x, cad.pdf(x), label="CAD")
    axes.plot(x, cad.c * cad.envelope.pdf(x), label="envelope")
    axes.set_xlabel("$x$")
    axes.set_ylabel("$p(x)$")
    axes.legend()
    fig.savefig(RESULTS / "task3-cad.png")


def task_a():
    """Use the Cauchy distribution as an enveloping function.
    You then need a constant c such that
        g(x) ≤ c h(x), ∀x ∈ (−∞, ∞).
    Any solution to this equation is correct, but a reasonably small value of c
    will produce a more efficient generator.
    In the present case, we already know an enveloping function, since g(x) ≤ 1 h(x).
    Thus c = 1/z is an obvious solution. In the acceptance condition z r c h(x_T) ≤ g(x_T),
    the variable z then cancels, thus the normalizing constant
    z of g(x) does not need to be known explicitely for the rejection method here.
    """
    c = 1 / g.normalization
    fig = plot_g(c, envelope=scipy.stats.cauchy)
    fig.savefig(RESULTS / "task3a.png")


def task_b():
    """Alternate envelope: Choose an alternate enveloping function for the rejec-
    tion method, e.g. constant for |x| < x0 and decreasing like 1/x2 for |x| ≥ x0 ,
    with a suitably chosen value for x0 . Depending on your choice of envelo-
    ping function, you may need to choose further constants. Note that your
    parameters do not need to be optimal. The envelope can be noticeably lar-
    ger than the desired distribution and still be quite good.
    """
    c = cad.normalization
    fig = plot_g(c, envelope=cad)
    fig.savefig(RESULTS / "task3b.png")


def main():
    """Run entire task3"""
    verify_cad()
    task_a()
    task_b()
    plt.show()
