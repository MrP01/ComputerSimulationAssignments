#include "Landscape.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: ./salesman_run [T0] [q] [NUMBER_OF_SWEEPS]" << std::endl;
    return 1;
  }

  double T0 = atol(argv[1]);
  double q = atof(argv[2]);
  size_t iterations = atol(argv[3]);

  srand(time(NULL));
  Landscape landscape;
  landscape.loadCityPositions();
  landscape.precomputeDistances();
  landscape.initState(T0, q);
  landscape.mcmcSimulate(iterations);
  return 0;
}
