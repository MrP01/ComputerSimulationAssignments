#pragma once
#include <assert.h>
#include <fstream>
#include <iostream>
#define NUMBER_OF_CITIES 22

typedef size_t TraversalConfiguration[NUMBER_OF_CITIES];

class Landscape {
 protected:
  double q, T0;
  double temperature;
  TraversalConfiguration state;
  double cityPositions[NUMBER_OF_CITIES][2];
  double cityDistances[NUMBER_OF_CITIES][NUMBER_OF_CITIES]; // symmetric matrix

 public:
  Landscape() = default;
  void initState(double T0, double q);
  void loadCityPositions();
  void precomputeDistances();
  double getEnergy(TraversalConfiguration order);
  double proposeNewTraversal(TraversalConfiguration &new_);
  void mcmcSweep(size_t steps, bool print_raw = true);
  void mcmcSimulate(size_t iterations);
  void nextTemperature(size_t j);
};
