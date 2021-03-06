#include "GA.hpp"
#include <limits>
#include <cmath>

#define THREAD_NUM 32

using namespace std;

default_random_engine generator;

// 比较两条染色体适应度，升序排序
int cmp(const ChrFit& chf1, const ChrFit& chf2){
    return chf1.fitness < chf2.fitness;
}

// 产生一条染色体，通过初始值突变完成
uint32_t generateChromosome() {
    uint32_t chr = 274922754;
    return chromosomeMutation(chr);
}

// 打印单条染色体上的基因
void printChromosome(uint32_t chr) {
    unsigned char* temp = (unsigned char*)&chr;
    for(int i = 0; i < 4; i++){
        if(i){
            printf(", ");
        }else{
            printf("[");
        }
        printf("%d", temp[i]);
    }
    printf("]\n");
}

// 染色体交叉，一条保留前面k个基因，另一条保留后面4-k个基因
uint32_t chromosomeCrossover(uint32_t chr1, uint32_t chr2) {
    uint32_t temp = generator() % 4;
    temp = (1 << (temp * 8))-1;
//    putchar('*');
//    printChromosome((chr1 & temp) | (chr2 & (UINT32_MAX-temp)));
    return (chr1 & temp) | (chr2 & (UINT32_MAX-temp));
}

// 染色体变异，单个基因的某一位翻转
uint32_t chromosomeMutation(uint32_t chr) {
    uint32_t temp = generator() % 32;
    temp = 1 << temp;
//    putchar('#');
//    printChromosome(chr ^ temp);
    return chr ^ temp;
}

// 使用两条染色体初始化权重，进行黑白棋对抗
int chromosomeCompetition(uint32_t chr1, uint32_t chr2) {
    int idx;
    bool round;
    Reversi game;
    AI bot;
    game.Initialize();
    round = true;
    while(!game.IsOver()){
        if(game.GetAvailable()){
            if(round){
                game.SetWeights((unsigned char*)&chr1);
            }else{
                game.SetWeights((unsigned char*)&chr2);
            }
            idx = bot.Search(game);
            game.PlacePiece(idx);
        }else{
            game.Skip();
        }
        round = !round;
    }
    return (!round) ^ (game.Evaluate() > 0);
}

// 产生一个特定大小的族群
uint32_t* GA::generateGroup(int size){
    uint32_t* group = new uint32_t[size];
    for(int i = 0; i < size; i++){
        group[i] = generateChromosome();
    }
    return group;
}

// 销毁一个特定大小的族群
void GA::destroyGroup(uint32_t *group) {
    delete[] group;
}

void* threadFunc(void* param){
    threadParam* par = (threadParam*)param;
    int step = par->size / par->threadCount;
    int start = par->threadIdx * step, end = (par->threadIdx+1) * step;
    int temp;
    for(int i = start; i < end; i++){
        for(int j = 0; j < par->size; j++){
            if(i == j){
                continue;
            }
            if(chromosomeCompetition(par->group[i], par->group[j])){
                temp += 1;
            }
            if(!chromosomeCompetition(par->group[j], par->group[i])){
                temp += 1;
            }
            par->result[i].fitness += temp;
        }
    }
}

// 计算一个族群中每个个体的适应度
ChrFit* GA::calculateFitness(uint32_t *group, int size) {
    printf("calculateFitness\n");
    ChrFit* result = new ChrFit[size];
    pthread_t* threads = new pthread_t[THREAD_NUM];
    threadParam* params = new threadParam[THREAD_NUM];
    for(int i = 0; i < size; i++){
        result[i].chr = group[i];
    }
    for(int i = 0; i < THREAD_NUM; i ++){
        params[i].init(group, result, size, THREAD_NUM, i);
        pthread_create(threads+i, nullptr, threadFunc, params+i);
    }
    for(int i = 0; i < THREAD_NUM; i++){
        pthread_join(threads[i], nullptr);
    }
    delete[] threads;
    delete[] params;
    sort(result, result+size, cmp);
    return result;
}

// 族群自然选择，将族群中适应度最高的部分族群放到族群后部，以便保留
void GA::doReproduction(ChrFit *groupFit, uint32_t* group, int size) {
    printf("doReproduction\n");
    for(int i = size-1; i >= 0; i--){
        group[i] = groupFit[i].chr;
    }
}

// 用族群中其他染色体交叉得到的新染色体代替未在自然选择中被保留的染色体
void GA::doCrossover(ChrFit *groupFit, uint32_t *group, int size) {
    printf("doCrossover\n");
    int index;
    uint32_t chr1, chr2;

    for(int i = 0; i < size; i++){
        index = (int)sqrt(generator() % (size * size));
        chr1 = groupFit[index].chr;
        index = (int)sqrt(generator() % (size * size));
        chr2 = groupFit[index].chr;
        group[i] = chromosomeCrossover(chr1, chr2);
    }
    delete[] groupFit;
}

// 对交叉产生的新染色体进行变异
void GA::doMutation(uint32_t *group, int size) {
    printf("doMutation\n");
    int mutationFactor = 2, index;
    for(int i = 0; i < size / mutationFactor; i++){
        index = (int)(generator() % size);
        group[index] = chromosomeMutation(group[index]);
    }
}

void GA::algorithm() {
    int epoch = 1000, size = 64, chosen = 32, loss;
    uint32_t* group = generateGroup(size);

    for(int i = 0; i < epoch; i++){
        printf("Epoch:\t%d\n", i);
        ChrFit* grpFit = calculateFitness(group, size);     // 计算适应度
        loss = grpFit[size-1].fitness - grpFit[0].fitness;  // 这个Loss不太靠谱，看看就好
        printf("Loss:\t%d\n", loss);
//        if(loss < 5){
//            break;
//        }
        doReproduction(grpFit, group, size);                // 自然选择
        doCrossover(grpFit, group, size-chosen);            // 交叉繁衍
        doMutation(group, size-chosen);                     // 基因变异
        printChromosome(group[size-1]);                     // 处于优势的染色体
        printChromosome(group[0]);                          // 处于劣势的染色体
        fflush(stdout);
    }
    destroyGroup(group);
}