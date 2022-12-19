"""U4, Computer Simulations SS2022
Peter Waldert, 11820727
"""
import asyncio
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

RESULTS = pathlib.Path("results") / "u4"
positions = np.loadtxt("u4/city-positions.txt", delimiter=",")
semaphore = asyncio.Semaphore(os.cpu_count() or 4)


# pylint: disable=invalid-name
async def simulate(T0: float, q: float, iterations: int):
    """Runs the underlying C++ MCMC simulation and loads the results"""
    async with semaphore:
        print(f"Starting with {T0=}, {q=:.3f}, {iterations=} ... ")
        process = await asyncio.create_subprocess_exec(
            "./build/bin/salesman_run",
            str(T0),
            str(q),
            str(iterations),
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        print("... finished")
    logs = np.fromstring(stdout, sep=", ")
    return logs.reshape((iterations, 3))


async def compare_beta():
    """For different beta, compare"""
    # Result = await simulate(1, 1, 100)
    q_to_try = [0.7, 1, 1.3]

    fig = plt.figure()
    fig2 = plt.figure()
    axes1: plt.Axes = fig.add_subplot(2, 1, 1)
    axes2: plt.Axes = fig.add_subplot(2, 1, 2)
    axes3: plt.Axes = fig2.add_subplot(2, 1, 1)
    axes4: plt.Axes = fig2.add_subplot(2, 1, 2)
    for q, Result in zip(q_to_try, await asyncio.gather(*[simulate(1, q, 100) for q in q_to_try])):
        temp = Result[:, 0]
        E_avg = Result[:, 1]
        E_var = Result[:, 2]
        axes1.plot(1 / temp, E_avg, label=f"$q = {q}$")
        axes2.plot(1 / temp, E_var, label=f"$q = {q}$")
        axes3.plot(E_avg, label=f"$q = {q}$")
        axes4.plot(E_var, label=f"$q = {q}$")

    axes1.set_xlabel(r"$\beta$ / 1")
    axes1.set_ylabel("$E_{avg}$ / 1")
    axes1.set_xlim([0, 80])
    axes1.legend()
    axes2.set_xlabel(r"$\beta$ / 1")
    axes2.set_ylabel("$E_{var}$ / 1")
    axes2.set_xlim([0, 80])
    axes2.legend()
    axes3.set_xlabel("Iteration / 1")
    axes3.set_ylabel("$E_{avg}$ / 1")
    axes3.legend()
    axes4.set_xlabel("Iteration / 1")
    axes4.set_ylabel("$E_{var}$ / 1")
    axes4.legend()
    fig.savefig(RESULTS / "q-comparison-beta-plot.png")
    fig2.savefig(RESULTS / "q-comparison-history.png")


def plot_cities():
    """Show a map"""
    fig = plt.figure()
    axes: plt.Axes = fig.add_subplot(1, 1, 1)
    axes.scatter(positions[:, 0], positions[:, 1])
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title("Map of cities")
    fig.savefig(RESULTS / "map.png")


def main():
    """Run full Traveling Salesman problem"""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(compare_beta()))
    # plot_cities()
    plt.show()


if __name__ == "__main__":
    main()
