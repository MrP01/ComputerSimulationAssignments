#include "Simulator.h"

QLocale german(QLocale::German, QLocale::Germany);

void Simulator::buildUI() {
  assert(SWEEPS_PER_TEMPERATURE % SWEEPS_PER_BLIT == 0);

  citySeries->setName("Cities to visit");
  travelSeries->setName("Travelling route");

  QChart *chart = new QChart;
  chart->setTitle("Travelling Salesman");
  chart->addSeries(citySeries);
  chart->addSeries(travelSeries);
  chart->createDefaultAxes();
  chart->axes(Qt::Horizontal).first()->setRange(-0.1, 1.1);
  chart->axes(Qt::Vertical).first()->setRange(-0.1, 1.1);

  QChartView *chartView = new QChartView(this);
  chartView->setRenderHint(QPainter::Antialiasing);
  chartView->setChart(chart);

  T0Input->setValue(T0);
  T0Input->setSingleStep(0.01);
  qInput->setValue(q);
  qInput->setSingleStep(0.01);
  autoTempChange->setChecked(true);

  temperatureSlider->setMinimum(0);
  temperatureSlider->setMaximum(TEMPERATURE_SLIDER_STEPS);
  temperatureSlider->setValue(TEMPERATURE_SLIDER_STEPS);
  QObject::connect(temperatureSlider, &QSlider::sliderMoved, this, [=](int newValue) {
    // Tk = T0 * k^(-q)  <==>  Tk / T0 = k^(-q)  <==>  k = (Tk/T0)^(-1/q)
    double goalTemperature = (double)newValue / TEMPERATURE_SLIDER_STEPS;
    _step = round(std::pow(goalTemperature / T0, -1.0 / q)) * SWEEPS_PER_TEMPERATURE;
    temperature = goalTemperature;
    std::cout << "Temp overwrite: " << goalTemperature << " and step: " << _step << std::endl;
  });

  connect(controlButton, &QPushButton::clicked, [=]() {
    if (controlButton->text() == "Start")
      startSimulation();
    else
      stopSimulation();
  });
  connect(resetButton, &QPushButton::clicked, [=]() {
    if (controlButton->text() == "Pause")
      stopSimulation();
    travelSeries->clear();
    initState(T0Input->value(), qInput->value());
    temperatureSlider->setValue(TEMPERATURE_SLIDER_STEPS);
    _step = 0;
  });
  connect(nextTempButton, &QPushButton::clicked, [=]() {
    _step += SWEEPS_PER_TEMPERATURE;
    nextTemperature(_step / SWEEPS_PER_TEMPERATURE);
    temperatureSlider->setValue(temperature * TEMPERATURE_SLIDER_STEPS);
  });

  auto mainWidget = new QWidget(this);
  auto mainLayout = new QGridLayout(mainWidget);
  mainLayout->addWidget(chartView, 1, 0);
  mainLayout->addWidget(label, 2, 0);
  mainLayout->addWidget(temperatureSlider, 1, 1);
  auto settingsLayout = new QHBoxLayout();
  settingsLayout->addWidget(T0Input);
  settingsLayout->addWidget(qInput);
  settingsLayout->addWidget(autoTempChange);
  mainLayout->addLayout(settingsLayout, 0, 0, 1, 2);
  auto buttonLayout = new QHBoxLayout();
  buttonLayout->addWidget(controlButton);
  buttonLayout->addWidget(resetButton);
  buttonLayout->addWidget(nextTempButton);
  mainLayout->addLayout(buttonLayout, 3, 0, 1, 2);
  setCentralWidget(mainWidget);
  setWindowTitle("Travelling Salesman Simulator");

  QShortcut *closeShortcut = new QShortcut(Qt::CTRL | Qt::Key_W, this);
  QObject::connect(closeShortcut, &QShortcut::activated, this, [=]() { close(); });
}

void Simulator::loadCityPositions() {
  Landscape::loadCityPositions();
  for (size_t i = 0; i < NUMBER_OF_CITIES; i++) {
    // std::cout << landscape.cityPositions[i][0] << ", " << landscape.cityPositions[i][1] << std::endl;
    *citySeries << QPointF(cityPositions[i][0], cityPositions[i][1]);
  }
}

void Simulator::mapTraversal() {
  travelSeries->clear();
  for (size_t i = 0; i < NUMBER_OF_CITIES; i++)
    *travelSeries << QPointF(cityPositions[state[i]][0], cityPositions[state[i]][1]);
}

void Simulator::timerEvent(QTimerEvent *event) {
  mcmcSweep(SWEEPS_PER_BLIT, false);
  mapTraversal();
  double energy = getEnergy(state);
  label->setText(QString("Energy: %1 | Temperature: %2 | Step: %3").arg(energy).arg(temperature).arg(_step));
  _step += SWEEPS_PER_BLIT;

  if ((_step % SWEEPS_PER_TEMPERATURE) == 0 && autoTempChange->isChecked()) {
    nextTemperature(_step / SWEEPS_PER_TEMPERATURE);
    temperatureSlider->setValue(temperature * TEMPERATURE_SLIDER_STEPS);
  }
}

void Simulator::startSimulation() {
  std::cout << "Starting simulation with T0 = " << T0 << " and q = " << q << std::endl;
  controlButton->setText("Pause");
  T0Input->setDisabled(true);
  qInput->setDisabled(true);
  _timerId = startTimer(10);
}

void Simulator::stopSimulation() {
  killTimer(_timerId);
  controlButton->setText("Start");
  T0Input->setDisabled(false);
  qInput->setDisabled(false);
}
