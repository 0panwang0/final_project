cmake_minimum_required(VERSION 3.10)
project(ArtificialIntelligenceExperiment)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

find_package(Threads)
add_executable(Reversi main.cpp Reversi.hpp Reversi.cpp GA.hpp GA.cpp)
target_link_libraries (Reversi ${CMAKE_THREAD_LIBS_INIT})
