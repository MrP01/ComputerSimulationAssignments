add_executable(salesman_run run.cpp Landscape.cpp)
add_executable(salesman_gui gui.cpp Landscape.cpp Simulator.cpp)
target_link_libraries(salesman_gui PUBLIC Qt6::Widgets Qt6::Charts)
