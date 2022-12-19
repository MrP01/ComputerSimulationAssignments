#include "Landscape.h"
#include <iterator>
#include <math.h>
#include <string.h>

void Landscape::loadCityPositions() {
  std::ifstream csvFile("u4/city-positions.txt");
  if (!csvFile.good() || !csvFile.is_open())
    throw std::runtime_error("File opening problem");
  std::string xpos, ypos;
  size_t index = 0;
  while (true) {
    std::getline(csvFile, xpos, ','); // break at comma
    if (!std::getline(csvFile, ypos)) // break at newline
      break;
    cityPositions[index][0] = std::stod(xpos);
    cityPositions[index][1] = std::stod(ypos);
    index++;
  }
  assert(index == NUMBER_OF_CITIES);
}

void Landscape::precomputeDistances() {
  for (size_t cityA = 0; cityA < NUMBER_OF_CITIES; cityA++) {
    for (size_t cityB = 0; cityB < NUMBER_OF_CITIES; cityB++) {
      double dx = cityPositions[cityA][0] - cityPositions[cityB][0];
      double dy = cityPositions[cityA][1] - cityPositions[cityB][1];
      cityDistances[cityA][cityB] = std::sqrt(dx * dx + dy * dy);
      // std::cout << "Distance from " << cityA << " to " << cityB << ": " << cityDistances[cityA][cityB] << std::endl;
    }
  }
}

void Landscape::initState(double T0_, double q_) {
  T0 = T0_;
  q = q_;
  temperature = T0_;
  for (size_t i = 0; i < NUMBER_OF_CITIES; i++)
    state[i] = i;
}

double Landscape::getEnergy(TraversalConfiguration order) {
  double totalDistanceTravelled = 0;
  for (size_t i = 1; i < NUMBER_OF_CITIES; i++) {
    totalDistanceTravelled += cityDistances[order[i - 1]][order[i]];
    // std::cout << order[i - 1] << "->";
  }
  totalDistanceTravelled += cityDistances[order[NUMBER_OF_CITIES - 1]][order[0]];
  // std::cout << std::endl;
  return totalDistanceTravelled;
}

double Landscape::proposeNewTraversal(TraversalConfiguration &new_) {
  memcpy(new_, state, NUMBER_OF_CITIES * sizeof(size_t)); // copy current configuration to new one
  size_t cityA_ = rand() % NUMBER_OF_CITIES;
  size_t cityB_ = rand() % NUMBER_OF_CITIES;
  size_t cityA = std::min(cityA_, cityB_);
  size_t cityB = std::max(cityA_, cityB_);
  for (size_t i = 0; i <= cityB - cityA; i++)
    new_[cityA + i] = state[cityB - i];
  // returns the energy delta
  if (cityA < cityB && cityA != 0 && cityB != NUMBER_OF_CITIES - 1)
    return cityDistances[state[cityA - 1]][state[cityB]] + cityDistances[state[cityA]][state[cityB + 1]] -
           cityDistances[state[cityA - 1]][state[cityA]] - cityDistances[state[cityB]][state[cityB + 1]];
  return getEnergy(new_) - getEnergy(state);
}

void Landscape::mcmcSweep(size_t steps, bool print_raw) {
  double E_sum = 0, E_squared_sum = 0;
  for (size_t i = 0; i < steps; i++) {
    TraversalConfiguration proposal;
    double deltaE = proposeNewTraversal(proposal);
    double acceptanceProbability = std::min(1.0, std::exp(-deltaE / temperature));
    if (((double)rand() / RAND_MAX) < acceptanceProbability)
      memcpy(state, proposal, NUMBER_OF_CITIES * sizeof(size_t));
    // std::cout << " -> " << acceptanceProbability << std::endl;
    double E = getEnergy(state);
    E_sum += E;
    E_squared_sum += E * E;
  }
  double E_avg = E_sum / steps;
  double E_var = E_squared_sum / steps - E_avg * E_avg;
  if (print_raw)
    std::cout << temperature << ", " << E_avg << ", " << E_var << ", ";
  else {
    std::cout << "Average energy for T = " << temperature << ": " << E_avg << std::endl;
    std::cout << "Variance for       T = " << temperature << ": " << E_var << std::endl;
  }
}

void Landscape::mcmcSimulate(size_t iterations) {
  for (size_t j = 0; j < iterations; j++) {
    mcmcSweep(NUMBER_OF_CITIES * NUMBER_OF_CITIES);
    nextTemperature(j);
  }
}

void Landscape::nextTemperature(size_t j) {
  temperature = T0 * std::pow(j + 1, -q);
  // std::cout << "New temp: " << temperature << std::endl;
}
