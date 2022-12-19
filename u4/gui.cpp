#include "Landscape.h"
#include "Simulator.h"

int main(int argc, char **argv) {
  QApplication app(argc, argv);
  setlocale(LC_NUMERIC, "en_US.UTF-8");
  srand(time(NULL));

  Simulator simulator;
  simulator.loadCityPositions();
  simulator.precomputeDistances();
  simulator.initState(1, 1);
  simulator.buildUI();
  simulator.resize(640, 480);
  simulator.show();
  return app.exec();
}
