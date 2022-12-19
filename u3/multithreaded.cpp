#include "Grid.h"
#include <pthread.h>
#include <thread>
#include <vector>

bool _stop_simulation = false;

struct Result {
  double average_e = 0;
  double average_m = 0;
  // std::vector<double> log;
};
struct ThreadArguments {
  size_t n_averages = 10000;
  size_t n_sweeps = 100;
  struct Result result;
};

void simulate(struct ThreadArguments *args) {
  Grid grid(5, 5, 0.4);
  int n_averages = args->n_averages;
  std::cout << "Start with n_averages = " << n_averages << std::endl;
  struct Result result;
  for (size_t n = 0; n < n_averages; n++) {
    grid.init_random_spins();
    grid.mcmc_simulate(args->n_sweeps, false);
    result.average_e += grid.get_energy_per_site();
    result.average_m += grid.get_magnetization_per_site();
  }
  result.average_e = result.average_e / n_averages;
  result.average_m = result.average_e / n_averages;
  args->result = result;
  std::cout << "End" << std::endl;
}

int main() {
  srand(time(NULL));
  size_t concurrency = std::thread::hardware_concurrency();
  // size_t concurrency = 1;
  pthread_t threads[concurrency];
  struct ThreadArguments thread_arguments[concurrency];
  for (size_t t = 0; t < concurrency; t++) {
    std::cout << "Starting thread " << t << std::endl;
    pthread_create(&threads[t], NULL, (void *(*)(void *))simulate, &thread_arguments[t]);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(t, &cpuset);
    pthread_setaffinity_np(threads[t], sizeof(cpuset), &cpuset);
  }
  struct Result overall_result;
  for (size_t t = 0; t < concurrency; t++) {
    std::cout << "Waiting for thread " << t << std::endl;
    pthread_join(threads[t], NULL);
    struct Result result = thread_arguments[t].result;
    overall_result.average_e += result.average_e;
    overall_result.average_m += result.average_m;
  }
  overall_result.average_e /= concurrency;
  overall_result.average_m /= concurrency;

  std::cout << "The average energy per site is " << overall_result.average_e << std::endl;
  std::cout << "The average magnetization per site is " << overall_result.average_m << std::endl;
  return 0;
}
