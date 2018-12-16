cmake_minimum_required(VERSION 3.10)
project(ArtificialIntelligenceExperiment)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

add_executable(Reversi main.cpp Reversi.hpp Reversi.cpp GA.hpp GA.cpp)

add_executable(Test GA.hpp GA.cpp Reversi.hpp Reversi.cpp test.cpp)