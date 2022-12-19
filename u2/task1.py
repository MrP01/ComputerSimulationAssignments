"""Assignment 2 - Computer Simulations SS2022
by Peter Waldert (11820727)
"""
import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.stats
import tqdm

RESULTS = pathlib.Path("results") / "u2"
# pylint: disable=protected-access
DistributionType = typing.Union[scipy.stats.rv_continuous, scipy.stats._distn_infrastructure.rv_frozen]  # type: ignore


def bayesian_expectation_std(occurences: np.ndarray, bins: int):
    """Given the raw occurences (heights) from the data, what are p_i and sigma_i?
    Based on the Bayesian formula (A.14, A.15 in lecture notes)"""
    num = sum(occurences)
    expectation_values = (occurences + 1) / (num + bins + 1)
    expected_stds = np.sqrt(expectation_values * (1 - expectation_values) / (num + bins + 2))
    return expectation_values, expected_stds


def plot_bayes_histo(axes: plt.Axes, samples: np.ndarray, dist: DistributionType, bins: int):
    """Use the Bayesian approach"""
    min_x, max_x = min(samples), max(samples)
    heights, edges = np.histogram(samples, bins=bins, range=(min_x, max_x))
    bincenters = 0.5 * (edges[1:] + edges[:-1])
    bin_width = (max_x - min_x) / bins
    x = np.linspace(min_x, max_x, 100)
    expectation_values, expected_stds = bayesian_expectation_std(heights, bins)
    axes.bar(bincenters, expectation_values / bin_width, width=bin_width * 0.9, yerr=expected_stds, label="histogram")
    axes.plot(x, dist.pdf(x), label="pdf", c="orange")
    axes.set_xlabel(r"$x$")
    axes.set_ylabel(r"Occurence $p(x)$")


def empirical_autocorrelation_function(x: np.ndarray, t: int):
    """Given a timeseries x, calculate rho^E(t) according to formula 1.38"""
    y = x[t:]
    x = x[: len(x) - t]
    # assert len(x) == len(y)
    x_avg, y_avg = x.mean(), y.mean()
    return sum((x - x_avg) * (y - y_avg)) / np.sqrt(sum((x - x_avg) ** 2) * sum((y - y_avg) ** 2))


def optimize_dx(xi: float, dx_guess: float):
    """Find the optimal dx for the given xi, such that the acceptance ratio is ~0.5"""
    target_distribution = TargetDistribution(xi=xi, name="target")
    for optimization in range(1, 13):
        mcmc = MCMC(target_distribution, 0, dx=dx_guess)
        mcmc.simulate(600)
        print(f"Acceptance ratio: {mcmc.acceptance_ratio()}")
        if mcmc.acceptance_ratio() > 0.5:
            dx_guess += 8 / optimization
            print("--> increasing dx: ", dx_guess)
        else:
            dx_guess -= 8 / optimization
            print("--> decreasing dx: ", dx_guess)
    return dx_guess


def binning_analysis(trace: np.ndarray, k: int):
    """Given the number `k`, calculate the variance sigma_k^2/N_Bk"""
    number_of_blocks = len(trace) // k
    averages = np.array([trace[i * k : i * k + k].mean() for i in range(number_of_blocks)])
    return sum((averages - averages.mean()) ** 2) / (number_of_blocks - 1) / number_of_blocks


class MCMC:
    """Markov Chain Monte Carlo using the Metropolis-Hastings algorithm"""

    def __init__(self, distribution: scipy.stats.rv_continuous, initial_state, dx) -> None:
        self.distribution = distribution
        self.state: float = initial_state
        self.accepted_proposals = 0
        self.rejected_proposals = 0
        self.dx = dx

    def proposal(self) -> float:
        """Returns a proposal for the next state x"""
        r = np.random.uniform()
        return self.state + (r - 0.5) * self.dx

    def update_state(self):
        """Perform one iteration of the Metropolis-Hastings algorithm"""
        proposal = self.proposal()
        proposal_probability = 1
        inverse_proposal_probability = proposal_probability
        acceptance_probability = min(
            1,
            (self.distribution.pdf(proposal) * inverse_proposal_probability)
            / (self.distribution.pdf(self.state) * proposal_probability),
        )
        # print(f"State Update: new proposal {proposal} with acceptance prob {acceptance_probability}")
        if np.random.uniform() < acceptance_probability:
            self.state = proposal
            self.accepted_proposals += 1
        else:
            self.rejected_proposals += 1

    def simulate(self, steps: int) -> np.ndarray:
        """Perform `steps` iterations of the algorithm and store the trace"""
        trace = []
        for _ in tqdm.tqdm(range(steps)):
            self.update_state()
            trace.append(self.state)
        return np.array(trace)

    def acceptance_ratio(self):
        """Calculates the acceptance ratio"""
        return self.accepted_proposals / (self.accepted_proposals + self.rejected_proposals)


class TargetDistribution(scipy.stats.rv_continuous):
    """The sought-after target distribution in this example"""

    def __init__(self, xi, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xi = xi

    def _pdf(self, x, *_):
        """Returns the PDF of our target"""
        phi_left = scipy.stats.norm(-self.xi)
        phi_right = scipy.stats.norm(self.xi)
        return (phi_left.pdf(x) + phi_right.pdf(x)) / 2


emp_autocorr_func = np.vectorize(empirical_autocorrelation_function, excluded=(0,))
binning = np.vectorize(binning_analysis, excluded=(0,))


def main():
    """Run the full simulation, analysis and plot results"""
    # optimize_dx(xi=2, dx_guess=10)
    total_samples = 1000
    clock = np.arange(total_samples)
    xis = (0, 2, 6)
    dxs = (6, 11, 45)
    for xi, dx in zip(xis, dxs):
        target_distribution = TargetDistribution(xi=xi, name="target")
        mcmc = MCMC(target_distribution, 0, dx=dx)
        trace = mcmc.simulate(total_samples)
        print(f"Acceptance ratio: {mcmc.acceptance_ratio()}")
        print(f"Subtask b) Value Average: {trace.mean()} +- {trace.std()}")

        t_max = total_samples // 20
        autocorrelation = emp_autocorr_func(trace, clock[:t_max])
        k = np.logspace(0, np.log2(total_samples) - 1, base=2, num=300).round().astype(int)
        sigma_sq_k = binning(trace, k)

        fig = plt.figure(figsize=(12, 8))
        axes: plt.Axes = fig.add_subplot(2, 2, 1)
        plot_bayes_histo(axes, trace, target_distribution, bins=20)
        axes: plt.Axes = fig.add_subplot(2, 2, 2)
        axes.scatter(clock, trace)
        axes.set_xlabel(r"Iteration $t$")
        axes.set_ylabel(r"State $x$")
        axes: plt.Axes = fig.add_subplot(2, 2, 3)
        axes.semilogy(abs(autocorrelation))
        axes.set_xlabel(r"Iteration $t$")
        axes.set_ylabel(r"Autocorrelation $\rho^E(t)$")
        axes: plt.Axes = fig.add_subplot(2, 2, 4)
        axes.semilogx(k, sigma_sq_k)
        axes.set_xlabel(r"$k$")
        axes.set_ylabel(r"Binning $\frac{\sigma_k^2}{N_{B,k}}$")
        fig.savefig(RESULTS / f"xi{xi}.png")

        print(
            f"Subtask e) Ratio between s(k={k[100]}) = {sigma_sq_k[100]} and "
            f"s(k=1) = {sigma_sq_k[0]}: {sigma_sq_k[100] / sigma_sq_k[0]}"
        )
    plt.show()


if __name__ == "__main__":
    main()
