#include <array>
#include <cmath>
#include <iostream>
#include <stdlib.h>

// be careful to set numeric values as floats here
#define PARTICLES 200              // number of particles
#define LJ_EPSILON 1.0             // Lennard-Jones energy
#define LJ_SIGMA 1.0               // equilibrium distance, 3.4 Angstrom
#define PARTICLE_MASS 1.0          // mass of a particle
#define LJ_CUTOFF_DISTANCE 0.1     // LJ explodes for very close particles, stop earlier
#define BOX_WIDTH 16 * LJ_SIGMA    // width of the infinitely high box
#define GRAVITY 8.532e1            // 9.81 m/s², actual value in reduced units: 8.532e-05
#define TAU 8.0e-4                 // time step
#define HEIGHT_HISTOGRAM_BINS 16   // into how many height boxes we aggregate particles
#define VELOCITY_HISTOGRAM_BINS 16 // similarly, number of bins for the velocity histogram
#define ONE_SECOND 2.1257e-12      // one second in reduced time unit
#define PLOT_HEIGHT 30             // height of the plot

#define square(x) (x * x)

#define LJ_CUT_DIST_SQ (LJ_CUTOFF_DISTANCE * LJ_CUTOFF_DISTANCE)
#define LJ_SIGMA_SQ (LJ_SIGMA * LJ_SIGMA)

using ParticleVectors = double (&)[PARTICLES][2];

struct HeightHistogram {
  double min, max;
  size_t heights[HEIGHT_HISTOGRAM_BINS] = {0};
  size_t maxHeight = 1;
};

struct VelocityHistogram {
  double min, max;
  size_t heights[VELOCITY_HISTOGRAM_BINS] = {0};
  size_t maxHeight = 1;
};

class ParticleBox {
 protected:
  double positions[PARTICLES][2];
  double velocities[PARTICLES][2];
  struct HeightHistogram heightHist;
  struct VelocityHistogram velocityHist;
  double totalMeanVelocity = 0;

  double distanceBetween(size_t i, size_t j) {
    return std::hypot(positions[i][0] - positions[j][0], positions[i][1] - positions[j][1]);
  }

 public:
  ParticleBox() = default;
  void initRandomly(double initialKineticEnergy, double initialGravitationalPotential);
  void simulate(size_t timesteps);
  void f(ParticleVectors &accelerations);
  void reflectParticles();
  double getKineticEnergy();
  double getGravitationalPotential();
  double getLJPotential();
  double getTotalEnergy();
  void computeHeightHistogram();
  void computeVelocityHistogram();
  void exportToCSV();
};
