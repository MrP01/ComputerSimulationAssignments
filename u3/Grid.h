#include <iostream>
#include <math.h>
#include <unistd.h>
#include <vector>

#define LATTICE_DIM1 lattice_dim1
#define LATTICE_DIM2 lattice_dim2
#define RESET "\033[0m"
#define GREEN "\033[32m" /* Green */
#define BLUE "\033[34m"  /* Blue */
#define SPIN_UP_CHAR "\033[42m "
#define SPIN_DOWN_CHAR "\033[44m "
#define flippyflip(spin) (spin == SPIN_UP) ? SPIN_DOWN : SPIN_UP

typedef enum SpinEnum { SPIN_UP, SPIN_DOWN } Spin;
extern bool _stop_simulation;

class Grid {
 private:
  size_t lattice_dim1;
  size_t lattice_dim2;
  // Spin spins[LATTICE_DIM1][LATTICE_DIM2];
  std::vector<std::vector<Spin>> spins;
  double beta = 1;
  double precomputed_probabilities[5];

  bool log = false;

 public:
  Grid(size_t dim1, size_t dim2, double beta);
  void init_random_spins();
  void print_me();
  void enable_logging() { log = true; };
  unsigned short get_number_of_aligned_spins(size_t i, size_t j);
  void mcmc_sweep();
  void mcmc_simulate(size_t iterations, bool animate = false, size_t animation_skip = 0, useconds_t delay = 0);
  void single_spin_flip(size_t i, size_t j);
  double get_magnetization_per_site();
  double get_energy_per_site();
  void save_logs_to_file(std::string filename);
};
