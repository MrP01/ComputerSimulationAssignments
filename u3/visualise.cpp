#include "Grid.h"
#include <iostream>
#include <signal.h>

bool _stop_simulation = false;
void sigint_handler(int signum) { _stop_simulation = true; }

void print_greeting(size_t Lx, size_t Ly) {
  std::cout << "--- The Ising Model of Magnetism on a " << Lx << " x " << Ly << " grid ---" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: ./ising_visualise [L_x] [L_y] [BETA] [NUMBER_OF_SWEEPS] [ANIMATE]" << std::endl;
    return 1;
  }

  size_t Lx = atol(argv[1]);
  size_t Ly = atol(argv[2]);
  double beta = atof(argv[3]);
  size_t iterations = atol(argv[4]);
  bool animate = false;
  if (argc >= 6)
    animate = std::string(argv[5]) == "true";

  srand(time(NULL));
  if (animate)
    if (std::system("clear") != 0)
      return 1;
  print_greeting(Lx, Ly);

  Grid grid(Ly, Lx, beta);
  grid.init_random_spins();

  signal(SIGINT, sigint_handler);
  clock_t start = clock();
  grid.mcmc_simulate(iterations, animate, 1, 1e5);
  if (animate) {
    if (std::system("clear") != 0)
      return 1;
    print_greeting(Lx, Ly);
    grid.print_me();
  }
  std::cout << "Elapsed time: " << (double)(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;

  std::cout << "--- Results: ---" << std::endl;
  std::cout << "The energy per site of our grid: " << grid.get_energy_per_site() << std::endl;
  std::cout << "The magnetization per site of our grid: " << grid.get_magnetization_per_site() << std::endl;
  return 0;
}
