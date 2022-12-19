#include "Landscape.h"
#include <QApplication>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QLabel>
#include <QMainWindow>
#include <QPushButton>
#include <QShortcut>
#include <QSlider>
#include <QTimer>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>

#define SWEEPS_PER_TEMPERATURE (NUMBER_OF_CITIES * NUMBER_OF_CITIES)
#define SWEEPS_PER_BLIT 11 // blit is one frame, or step
#define TEMPERATURE_SLIDER_STEPS 500

class Simulator : public Landscape, public QMainWindow {
 private:
  size_t _step = 0;
  size_t _timerId;
  QPushButton *controlButton = new QPushButton("Start");
  QPushButton *resetButton = new QPushButton("Reset");
  QPushButton *nextTempButton = new QPushButton("Next Temperature");
  QCheckBox *autoTempChange = new QCheckBox("Auto-Temperature");
  QDoubleSpinBox *T0Input = new QDoubleSpinBox();
  QDoubleSpinBox *qInput = new QDoubleSpinBox();
  QScatterSeries *citySeries = new QScatterSeries();
  QLineSeries *travelSeries = new QLineSeries();
  QLabel *label = new QLabel("Travelling Salesman Simulator");
  QSlider *temperatureSlider = new QSlider(Qt::Vertical);
  void mapTraversal();
  void timerEvent(QTimerEvent *event);

 public:
  Simulator() = default;
  void buildUI();
  void loadCityPositions();
  void startSimulation();
  void stopSimulation();
};
