//
// Created by James on 2018/12/16.
//
#ifndef ARTIFICIALINTELLIGENCEEXPERIMENT_GA_HPP
#define ARTIFICIALINTELLIGENCEEXPERIMENT_GA_HPP

#include <random>
#include "Reversi.hpp"

typedef struct chromosomeFitness{
    uint32_t chr;
    int fitness;
    chromosomeFitness():fitness(0){};
} ChrFit;

class GA {
public:
    uint32_t* generateGroup(int size);
    void destroyGroup(uint32_t* group);
    ChrFit* calculateFitness(uint32_t* group, int size);
    void doReproduction(ChrFit* groupFit, uint32_t* group, int size, int choosen);
    void doCrossover(ChrFit* groupFit, uint32_t* group, int size);
    void doMutation(uint32_t* group, int size);
    void algorithm();
};


#endif //ARTIFICIALINTELLIGENCEEXPERIMENT_GA_HPP
