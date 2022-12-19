#include "Grid.h"
#include <iostream>

bool _stop_simulation = false;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: ./ising_run [L] [BETA] [NUMBER_OF_SWEEPS]" << std::endl;
    return 1;
  }

  size_t L = atol(argv[1]);
  double beta = atof(argv[2]);
  size_t iterations = atol(argv[3]);
  srand(time(NULL));
  Grid grid(L, L, beta);
  grid.enable_logging();
  grid.init_random_spins();
  grid.mcmc_simulate(iterations);
  return 0;
}
