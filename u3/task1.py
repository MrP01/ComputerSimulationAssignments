"""Computer Simulations Assignment 3, SS2022
Michael Obermayr and Peter Waldert
"""
import asyncio
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

RESULTS = pathlib.Path("results") / "u3"
semaphore = asyncio.Semaphore(os.cpu_count() or 4)


def autocorr(x, lags):
    """non partial numpy auto-correlation function, which also estimates timescale and true error"""
    var = np.var(x)
    if var == 0:
        var = 1  # ensure, that variance is not 0, kinda dumb
    xp = x - x.mean()
    corr = np.correlate(xp, xp, "full")[len(x) - 1 :] / var / len(x)
    corr = corr[: len(lags)]

    # find timescale via inverse slope, first determine end of log decrease
    index = 1  # fallback
    for index in range(np.size(corr) - 1):
        if corr[index] <= 0:
            break
        if corr[index + 1] / corr[index] <= 1e-2:
            break
    index = round(0.8 * index)  # decrease index for more stability

    series = np.log(corr[0:index])
    slope = np.polyfit(np.arange(0, index), series, 1)
    timescale = -1 / slope[0]

    # estimate true error
    true_var = var / len(x) * 2 * timescale
    true_error = np.sqrt(true_var)

    return corr, timescale, true_error


async def simulate(L: int, beta: float, iterations: int):
    """Runs the underlying C++ MCMC simulation and loads the results"""
    async with semaphore:
        print(f"Starting with {L=}, {beta=:.3f}, {iterations=} ... ")
        process = await asyncio.create_subprocess_exec(
            "./build/bin/ising_run",
            str(L),
            str(beta),
            str(iterations),
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        print("... finished")
    logs = np.fromstring(stdout, sep=", ")
    return logs.reshape((iterations, 2))


async def simulate_and_average(L: int, beta: float, iterations: int):
    """Returns averaged energy and absolute magnetization after 10% of the run"""
    logs = await simulate(L, beta, iterations)
    return (
        np.average(logs[round(0.1 * iterations) : -1, 0]),
        np.average(np.abs(logs[round(0.1 * iterations) : -1, 1])),
    )


async def plot_history():
    """Plot magnetization and energy history"""
    kwargs = {"L": 50, "beta": 0.4, "iterations": 1000}
    logs = await simulate(**kwargs)
    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(2, 1, 1)
    axes.plot(logs[:, 0])
    axes.set_xlabel("Iteration $t$")
    axes.set_ylabel("Energy $E$")
    axes.set_title(f"Simulation with {kwargs}")
    axes: plt.Axes = fig.add_subplot(2, 1, 2)
    axes.plot(logs[:, 1])
    axes.set_xlabel("Iteration $t$")
    axes.set_ylabel("Magnetization $M$")
    fig.savefig(RESULTS / "history.png")


async def compare_beta(L):
    """Task 5.2. Plot magnetization and average energy for several betas"""
    no_sweeps = 20000
    betas = np.linspace(0.2, 0.6, num=50)
    energies, abs_magnetizations = [], []
    for energy, amag in await asyncio.gather(*[simulate_and_average(L, beta, no_sweeps) for beta in betas.repeat(3)]):
        # thermalize cutout
        energies.append(energy)
        abs_magnetizations.append(amag)
    energies = np.array(energies).reshape((50, 3)).mean(axis=1)
    abs_magnetizations = np.array(abs_magnetizations).reshape((50, 3)).mean(axis=1)
    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(2, 1, 1)
    axes.plot(betas, energies)
    axes.set_xlabel(r"$\beta$")
    axes.set_ylabel("Energy $E$")
    axes.set_title(f"Simulation with different betas and Grid size {L}x{L}")
    axes: plt.Axes = fig.add_subplot(2, 1, 2)
    axes.plot(betas, abs_magnetizations)
    axes.set_xlabel(r"$\beta$")
    axes.set_ylabel("Magnetization $M$")
    fig.savefig(RESULTS / f"PhaseTransition_Grid{L}.png")


async def compare_grid_sizes():
    """Plot magnetization and energy history"""
    sizes = np.arange(1, 30)
    energies, magnetizations = [], []
    for logs in await asyncio.gather(*[simulate(L, 0.4, 8000) for L in sizes]):
        energy, mag = logs[-1, 0], logs[-1, 1]
        energies.append(energy)
        magnetizations.append(mag)
    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(2, 1, 1)
    axes.plot(sizes, energies)
    axes.set_xlabel("Grid size $L$")
    axes.set_ylabel("Energy $E$")
    axes.set_title("Simulation with different grid sizes")
    axes: plt.Axes = fig.add_subplot(2, 1, 2)
    axes.plot(sizes, magnetizations)
    axes.set_xlabel("Grid size $L$")
    axes.set_ylabel("Magnetization $M$")
    fig.savefig(RESULTS / "grid-sizes.png")


async def plot_autocorrelations():
    """Plot autocorrelations"""
    kwargs = {"L": 50, "beta": 0.4, "iterations": 1000}
    logs = await simulate(**kwargs)

    E_series = logs[:, 0]
    M_series = logs[:, 1]
    # calculate autocorrelation for E and M
    lags = np.arange(round(len(E_series) / 2))
    E_autocorr, E_timescale, E_error = autocorr(E_series, lags)
    M_autocorr, M_timescale, M_error = autocorr(M_series, lags)

    print(f"Average energy: {np.average(E_series):1.3f} +- {E_error:1.3f}")
    print(f"Average magnetisation: {np.average(M_series):1.3f} +- {M_error:1.3f}")

    # plot autocorrelations
    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(2, 1, 1)
    axes.plot(lags, E_autocorr)
    axes.set_yscale("log")
    axes.set_xlabel("Displacement")
    axes.set_ylabel("ACF for $E$")
    axes.set_title(
        "Autocorrelation coefficients for Energy and Magnetization \n "
        f"Timescale for $E$: {E_timescale:1.1f}, Timescale for $M$: {M_timescale:1.1f}"
    )
    axes: plt.Axes = fig.add_subplot(2, 1, 2)
    axes.plot(lags, M_autocorr)
    axes.set_yscale("log")
    axes.set_xlabel("Displacement")
    axes.set_ylabel("ACF for $M$")
    fig.savefig(RESULTS / "autocorrelations.png")


async def freeze_system():
    """Task 4.1., simulate for low temperatures at various grid sizes, with+without thermalization"""
    betas = np.array([0.2, 0.4, 0.6])
    sizes = np.array([4, 6, 8])
    energies, magnetizations = [], []

    no_sweeps = 8000

    # simulate
    for L in sizes:
        for logs in await asyncio.gather(*[simulate(L, beta, no_sweeps) for beta in betas]):
            energies.append(list(logs[:, 0]))
            magnetizations.append(list(logs[:, 1]))

    # calculate and plot autocorrelation + errors, combine plots for every beta
    b = len(betas)
    s = len(sizes)

    for i in range(s):
        for j in range(b):
            print(
                "==========Grid size: %dx%d, beta = %1.1f at %d sweeps=========="
                % (sizes[i], sizes[i], betas[j], no_sweeps)
            )
            E_series = np.array(energies[s * i + j])
            M_series = np.array(magnetizations[s * i + j])

            # without thermalization
            lags = np.arange(round(len(E_series) / 2))
            E_autocorr, E_timescale, E_error = autocorr(E_series, lags)
            M_autocorr, M_timescale, M_error = autocorr(M_series, lags)
            print(
                "Autocorrelation \n E = %1.3f +- %1.3f, tau = %1.1f \n M = %1.3f +- %1.3f, tau = %1.1f"
                % (np.average(E_series), E_error, E_timescale, np.average(M_series), M_error, M_timescale)
            )

            fig = plt.figure(figsize=(16, 8))
            fig.suptitle("Grid size: %dx%d, $\\beta$ = %1.1f" % (sizes[i], sizes[i], betas[j]))
            axes: plt.Axes = fig.add_subplot(2, 3, 1)
            axes.plot(E_series)
            axes.set_xlabel("Iteration $t$")
            axes.set_ylabel("Energy $E$")
            axes.set_title("Time series")
            axes: plt.Axes = fig.add_subplot(2, 3, 4)
            axes.plot(M_series)
            axes.set_xlabel("Iteration $t$")
            axes.set_ylabel("Magnetization $M$")

            axes: plt.Axes = fig.add_subplot(2, 3, 2)
            axes.plot(lags, E_autocorr)
            axes.set_yscale("log")
            axes.set_xlabel("Displacement")
            axes.set_ylabel("ACF for $E$")
            axes.set_title(
                "Autocorrelation \n E = %1.3f +- %1.3f, $\\tau$ = %1.1f \n M = %1.3f +- %1.3f, $\\tau$ = %1.1f"
                % (np.average(E_series), E_error, E_timescale, np.average(M_series), M_error, M_timescale)
            )
            axes: plt.Axes = fig.add_subplot(2, 3, 5)
            axes.plot(lags, M_autocorr)
            axes.set_yscale("log")
            axes.set_xlabel("Displacement")
            axes.set_ylabel("ACF for $M$")

            # with thermalization, cut out first 10 percent
            E_series_t = E_series[round(0.1 * len(E_series)) : -1]
            M_series_t = M_series[round(0.1 * len(M_series)) : -1]
            lags_t = np.arange(round(len(E_series_t) / 2))
            E_autocorr_t, E_timescale_t, E_error_t = autocorr(E_series_t, lags_t)
            M_autocorr_t, M_timescale_t, M_error_t = autocorr(M_series_t, lags_t)
            print(
                "Thermalized Autocorrelation \n E = %1.3f +- %1.3f, tau = %1.1f \n M = %1.3f +- %1.3f, tau = %1.1f"
                % (np.average(E_series_t), E_error_t, E_timescale_t, np.average(M_series_t), M_error_t, M_timescale_t)
            )

            axes: plt.Axes = fig.add_subplot(2, 3, 3)
            axes.plot(lags_t, E_autocorr_t)
            axes.set_yscale("log")
            axes.set_xlabel("Displacement")
            axes.set_ylabel("ACF for $E$")
            axes.set_title(
                "Thermalized Autocorrelation \n E = %1.3f +- %1.3f, $\\tau$ = %1.1f \n M = %1.3f +- %1.3f, $\\tau$ = %1.1f"
                % (np.average(E_series_t), E_error_t, E_timescale_t, np.average(M_series_t), M_error_t, M_timescale_t)
            )
            axes: plt.Axes = fig.add_subplot(2, 3, 6)
            axes.plot(lags_t, M_autocorr_t)
            axes.set_yscale("log")
            axes.set_xlabel("Displacement")
            axes.set_ylabel("ACF for $M$")
            pngname = "Systemfreeze_Grid%d_beta%1.1f.png" % (sizes[i], betas[j])
            fig.tight_layout()
            fig.savefig(RESULTS / pngname)


async def plot_critical_beta():
    """Task 4.2., simulate for critical temperatures at various grid sizes, with thermalization cutout"""
    beta = 0.44069
    sizes = np.array([4, 8, 16, 32])
    no_sweeps = 100 * np.square(sizes)  # be careful, takes extremely long for prefactor >100
    s = len(sizes)
    energies, magnetizations = [], []

    # simulate
    for logs in await asyncio.gather(*[simulate(size, beta, no_sweep) for size, no_sweep in zip(sizes, no_sweeps)]):
        energies.append(list(logs[:, 0]))
        magnetizations.append(list(logs[:, 1]))

    # calculate and plot autocorrelation + errors, combine plots for every beta
    for i in range(s):
        print(
            "==========Grid size: %dx%d, beta = %1.4f at %d sweeps==========" % (sizes[i], sizes[i], beta, no_sweeps[i])
        )
        E_series = np.array(energies[i])
        M_series = np.array(magnetizations[i])

        # plot time series
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle("Grid size: %dx%d, $\\beta_c$ = %1.4f" % (sizes[i], sizes[i], beta))
        axes: plt.Axes = fig.add_subplot(2, 2, 1)
        axes.plot(E_series)
        axes.set_xlabel("Iteration $t$")
        axes.set_ylabel("Energy $E$")
        axes.set_title("Time series")
        axes: plt.Axes = fig.add_subplot(2, 2, 3)
        axes.plot(M_series)
        axes.set_xlabel("Iteration $t$")
        axes.set_ylabel("Magnetization $M$")

        # with thermalization, cut out first 10 percent
        E_series_t = E_series[round(0.1 * len(E_series)) : -1]
        M_series_t = M_series[round(0.1 * len(M_series)) : -1]
        lags_t = np.arange(round(len(E_series_t) / 2))
        E_autocorr_t, E_timescale_t, E_error_t = autocorr(E_series_t, lags_t)
        M_autocorr_t, M_timescale_t, M_error_t = autocorr(M_series_t, lags_t)
        print(
            "Thermalized Autocorrelation \n E = %1.3f +- %1.3f, tau = %1.1f \n M = %1.3f +- %1.3f, tau = %1.1f"
            % (np.average(E_series_t), E_error_t, E_timescale_t, np.average(M_series_t), M_error_t, M_timescale_t)
        )

        axes: plt.Axes = fig.add_subplot(2, 2, 2)
        axes.plot(lags_t, E_autocorr_t)
        axes.set_yscale("log")
        axes.set_xlabel("Displacement")
        axes.set_ylabel("ACF for $E$")
        axes.set_title(
            "Thermalized Autocorrelation \n E = %1.3f +- %1.3f, $\\tau$ = %1.1f \n M = %1.3f +- %1.3f, $\\tau$ = %1.1f"
            % (np.average(E_series_t), E_error_t, E_timescale_t, np.average(M_series_t), M_error_t, M_timescale_t)
        )
        axes: plt.Axes = fig.add_subplot(2, 2, 4)
        axes.plot(lags_t, M_autocorr_t)
        axes.set_yscale("log")
        axes.set_xlabel("Displacement")
        axes.set_ylabel("ACF for $M$")
        pngname = "CriticalTemp_Grid%d.png" % (sizes[i])
        fig.tight_layout()
        fig.savefig(RESULTS / pngname)


def main():
    """Run everything"""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        asyncio.gather(
            # ====== program testing ======
            plot_history(),
            plot_autocorrelations(),
            compare_grid_sizes(),
            # ====== assignment tasks ======
            freeze_system(),
            plot_critical_beta(),
            compare_beta(8),
            compare_beta(32),
        )
    )


if __name__ == "__main__":
    main()
