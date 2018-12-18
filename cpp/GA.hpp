//
// Created by James on 2018/12/16.
//
#ifndef ARTIFICIALINTELLIGENCEEXPERIMENT_GA_HPP
#define ARTIFICIALINTELLIGENCEEXPERIMENT_GA_HPP

#include <random>
#include <pthread.h>
#include "Reversi.hpp"

// 存储染色体及其适应度
typedef struct chromosomeFitness{
    uint32_t chr;
    int fitness;
    chromosomeFitness():chr(0xFFFFFFFF), fitness(0){};
} ChrFit;

typedef struct threadParam{
    uint32_t* group;
    int* schedule;
    ChrFit* result;
    int size;
    int degree;
    int step;
    int index;
    threadParam(uint32_t* grp, int* sch, ChrFit* res, int sz, int deg, int st, int idx)
        :group(grp),schedule(sch),result(res),size(sz),degree(deg), step(st),index(idx){}
}threadParam;

uint32_t generateChromosome();      // 产生一条染色体
void printChromosome(uint32_t chr); // 打印染色体上的基因（权重）
uint32_t chromosomeCrossover(uint32_t chr1, uint32_t chr2);     // 染色体交叉
uint32_t chromosomeMutation(uint32_t chr);                      // 染色体突变
int chromosomeCompetition(uint32_t chr1, uint32_t chr2);    // 染色体竞争/对战

class GA {
public:
    uint32_t* generateGroup(int size);      // 产生族群
    void destroyGroup(uint32_t* group);     // 销毁族群
    ChrFit* calculateFitness(uint32_t* group, int size);    // 计算适应度（运算量的大头）
    void doReproduction(ChrFit* groupFit, uint32_t* group, int size, int choosen);  // 自然选择
    void doCrossover(ChrFit* groupFit, uint32_t* group, int size);                  // 交叉繁衍
    void doMutation(uint32_t* group, int size);                                     // 族群突变
    void algorithm();   // GA算法
};


#endif //ARTIFICIALINTELLIGENCEEXPERIMENT_GA_HPP
