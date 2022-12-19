"""U5, Computer Simulations SS2022
Peter Waldert, 11820727
"""
import pathlib
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import scipy.stats

RESULTS = pathlib.Path("results") / "u5"
TMP = pathlib.Path(tempfile.gettempdir())

SIGMA_IN_SI = 3.4 * scipy.constants.angstrom
M_IN_SI = 39 * scipy.constants.atomic_mass
EPSILON_IN_SI = 120 * scipy.constants.Boltzmann
G_IN_REDUCED = 9.81 * SIGMA_IN_SI * M_IN_SI / EPSILON_IN_SI
LJ_CUTOFF_DISTANCE = 0.5  # a tenth of Sigma
LJ_CUTOFF_FORCE = 24 / LJ_CUTOFF_DISTANCE**2 * (2 / LJ_CUTOFF_DISTANCE**12 - 1 / LJ_CUTOFF_DISTANCE**6)


def verify_bolzmann():
    """Given the simulation data, analyze"""
    print(30 * "-")
    print("Boltzmann Distribution:")
    velocities = np.loadtxt(TMP / "velocities.csv", delimiter=", ")
    abs_velocity = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)
    loc, scale = scipy.stats.maxwell.fit(abs_velocity)
    print(loc, scale)
    print("k_B * T / m =", scale**2)

    x = np.linspace(min(abs_velocity), max(abs_velocity), 100)

    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(1, 1, 1)
    axes.hist(abs_velocity, bins=30, density=True, label="True Data")
    axes.plot(x, scipy.stats.maxwell.pdf(x, loc, scale), label="PDF Fit")
    axes.set_xlabel("$|v|$")
    axes.set_ylabel("$p(|v|)$")
    axes.legend()
    fig.savefig(RESULTS / "bolzmann-fit.png")


def verify_barometric():
    """Check whether the barometric equation holds"""
    print(30 * "-")
    print("Barometric Equation:")
    positions = np.loadtxt(TMP / "positions.csv", delimiter=", ")

    bins = 10
    hist_heights, hist_edges = np.histogram(positions[:, 1], bins=bins)
    bincenters = 0.5 * (hist_edges[1:] + hist_edges[:-1])
    np.sqrt(hist_heights)

    x = bincenters
    y = hist_heights  # corresponds to densities
    linfit = np.polyfit(x, np.log(y), 1)
    x_theo = np.linspace(hist_edges[0], hist_edges[-1], 100)
    y_theo = np.exp(np.polyval(linfit, x_theo))
    h_s = -1 / linfit[0]  # ln(rho(h)) = ln(rho_0) - 1 / h_s * h
    g = 8.532e1  # actual used value
    print(f"{h_s = }, {g = }")
    print("k_B * T / m =", h_s * g)

    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(1, 1, 1)
    # , yerr=expected_stds
    axes.bar(bincenters, hist_heights, width=(hist_edges[-1] - hist_edges[0]) / bins, label="Histogram")
    axes.plot(x_theo, y_theo, label="Fit", c="orange")
    axes.set_xlabel(r"Height $h$ / $\overline{\sigma}$")
    axes.set_ylabel(r"$\rho(h)$")
    axes.legend()
    fig.savefig(RESULTS / "barometric-fit.png")


def plot_cutoff():
    """Plot the LJ force"""
    print(30 * "-")
    print("LJ Cutoff:")
    distance = np.linspace(0.1, 1, 1000)  # in reduced units
    distance_sq = distance**2
    lj_force = 24 / distance_sq * (2 / distance_sq**6 - 1 / distance_sq**3)
    print(f"Cutoff force for r < {LJ_CUTOFF_DISTANCE}: F_cutoff = {LJ_CUTOFF_FORCE}")
    lj_force[distance < LJ_CUTOFF_DISTANCE] = LJ_CUTOFF_FORCE

    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(1, 1, 1)
    axes.plot(distance, lj_force)
    axes.set_xlabel(r"$r \,/\, \overline{\sigma}$")
    axes.set_ylabel(r"$F_{LJ} \,/\, \overline{F}$")
    fig.savefig(RESULTS / "lj-force.png")


def main():
    """Run everything"""
    print("1 tu (time unit) in seconds:", SIGMA_IN_SI * np.sqrt(M_IN_SI / EPSILON_IN_SI))
    print("g in reduced units:", G_IN_REDUCED)
    print("k_B * T / m for Argon at room temp:", scipy.constants.Boltzmann * 300 / M_IN_SI)

    verify_bolzmann()
    verify_barometric()
    plot_cutoff()
    plt.show()


if __name__ == "__main__":
    main()
