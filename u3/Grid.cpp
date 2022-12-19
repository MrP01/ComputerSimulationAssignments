#include "Grid.h"
#include <fstream>

Grid::Grid(size_t dim1, size_t dim2, double beta) : lattice_dim1(dim1), lattice_dim2(dim2), beta(beta) {
  double normalization = 0;
  for (int n_aligned_spins = 0; n_aligned_spins <= 4; n_aligned_spins++) {
    // compute switching probability by dividing probabilities of inital and final state,
    // via formula prob  = exp(-beta * delta_Energy)
    //
    // delta_Energy = -2 * initial single site energy * 2, first factor from divison, second factor from counting total
    // energy (2* shared interaction energies)
    // single site energy = 2 - aligned neighbours, interaction energies are shared between neighbours
    precomputed_probabilities[n_aligned_spins] = exp(-beta * (n_aligned_spins - 2) * 4);
    // std::cout << "n_aligned = " << n_aligned_spins
    //           << " --> acceptance probability: " << precomputed_probabilities[n_aligned_spins] << std::endl;
  }
  for (size_t i = 0; i < lattice_dim1; i++)
    spins.push_back(std::vector<Spin>(lattice_dim2, SPIN_DOWN));
}

void Grid::init_random_spins() {
  for (size_t i = 0; i < LATTICE_DIM1; i++)
    for (size_t j = 0; j < LATTICE_DIM2; j++)
      spins[i][j] = (rand() % 2) ? SPIN_UP : SPIN_DOWN;
}

void Grid::print_me() {
  for (size_t i = 0; i < LATTICE_DIM1; i++) {
    for (size_t j = 0; j < LATTICE_DIM2; j++)
      std::cout << ((spins[i][j] == SPIN_UP) ? SPIN_UP_CHAR : SPIN_DOWN_CHAR);
    std::cout << RESET << std::endl;
  }
}

unsigned short Grid::get_number_of_aligned_spins(size_t i, size_t j) {
  Spin my_spin = spins[i][j];
  unsigned short n_aligned_spins = 0;
  n_aligned_spins += (my_spin == spins[(i - 1) % LATTICE_DIM1][j]);
  n_aligned_spins += (my_spin == spins[(i + 1) % LATTICE_DIM1][j]);
  n_aligned_spins += (my_spin == spins[i][(j - 1) % LATTICE_DIM2]);
  n_aligned_spins += (my_spin == spins[i][(j + 1) % LATTICE_DIM2]);
  return n_aligned_spins;
}

void Grid::single_spin_flip(size_t i, size_t j) {
  unsigned short n_aligned = get_number_of_aligned_spins(i, j);
  // if (n_aligned < 2) {
  //   spins[i][j] = flippyflip(spins[i][j]);
  //   return;
  // }
  double probability = precomputed_probabilities[n_aligned];
  double uniform = (double)rand() / RAND_MAX;
  if (uniform < probability) {
    // accept proposal
    spins[i][j] = flippyflip(spins[i][j]);
  }
}

void Grid::mcmc_sweep() {
  for (size_t n = 0; n < (LATTICE_DIM1 * LATTICE_DIM2); n++)
    single_spin_flip(rand() % LATTICE_DIM1, rand() % LATTICE_DIM2);
  if (log)
    std::cout << get_energy_per_site() << ", " << get_magnetization_per_site() << ", ";
}

void Grid::mcmc_simulate(size_t iterations, bool animate, size_t animation_skip, useconds_t delay) {
  if (animate)
    std::cout << "\033[s" << std::flush;
  for (size_t iteration = 0; iteration < iterations; iteration++) {
    mcmc_sweep();
    if (animate && iteration % (animation_skip + 1) == 0) {
      std::cout << "\033[u" << std::flush;
      print_me();
      std::cout << "Iteration: " << iteration << " / " << iterations
                << " | Magnetization per site: " << get_magnetization_per_site()
                << " | Energy per site: " << get_energy_per_site() << std::endl;
      usleep(delay);
    }
    if (_stop_simulation)
      break;
  }
}

double Grid::get_magnetization_per_site() {
  double result = 0;
  for (size_t i = 0; i < LATTICE_DIM1; i++)
    for (size_t j = 0; j < LATTICE_DIM2; j++)
      if (spins[i][j] == SPIN_UP)
        result++;
      else
        result--;
  return result / (LATTICE_DIM1 * LATTICE_DIM2);
}

double Grid::get_energy_per_site() {
  double result = 0;
  for (size_t i = 0; i < LATTICE_DIM1; i++)
    for (size_t j = 0; j < LATTICE_DIM2; j++) {
      int n_aligned = get_number_of_aligned_spins(i, j);
      // single site energy = aligned neighbours - 2, no factor 2 because interactions are shared
      result += -(n_aligned - 2);
    }
  return result / (LATTICE_DIM1 * LATTICE_DIM2);
}
