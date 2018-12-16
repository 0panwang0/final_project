#include "GA.hpp"
#include <limits>

using namespace std;

std::default_random_engine generator;

int cmp(const ChrFit& chf1, const ChrFit& chf2){
    return chf1.fitness < chf2.fitness;
}

uint32_t generateChromosome() {
    return generator() % UINT32_MAX;
}

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

uint32_t chromosomeCrossover(uint32_t chr1, uint32_t chr2) {
    uint32_t temp = generator() % 4;
    temp = (1 << (temp * 8))-1;
//    putchar('*');
//    printChromosome((chr1 & temp) | (chr2 & (UINT32_MAX-temp)));
    return (chr1 & temp) | (chr2 & (UINT32_MAX-temp));
}

uint32_t chromosomeMutation(uint32_t chr) {
    uint32_t temp = generator() % 32;
    temp = 1 << temp;
//    putchar('#');
//    printChromosome(chr ^ temp);
    return chr ^ temp;
}

int chromosomeCompetition(Reversi& game, AI& bot, uint32_t chr1, uint32_t chr2) {
    int idx;
    bool round;
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

uint32_t* GA::generateGroup(int size){
    uint32_t* group = new uint32_t[size];
    for(int i = 0; i < size; i++){
        group[i] = generateChromosome();
    }
    return group;
}

void GA::destroyGroup(uint32_t *group) {
    delete[] group;
}

ChrFit* GA::calculateFitness(uint32_t *group, int size) {
    printf("calculateFitness\n");
    ChrFit* result = new ChrFit[size];
    Reversi game;
    AI bot;
    int score, step, temp;
    for(int i = 0; i < size; i++){
        result[i].chr = group[i];
    }
//    for(int i = 0; i < size; i++){
//        for(int j = 0; j < size; j++){
//            if(i == j){
//                continue;
//            }
//            temp = chromosomeCompetition(game, bot, group[i], group[j]) ? 1 : -1;
//            result[i].fitness += temp;
//            result[j].fitness -= temp;
//        }
//    }
    int schedule[size];
    for(int i = 0; i < size; i++){
        schedule[i] = i;
    }
    step = 2;
    while(step <= size){
        for(int i = 0; i < size; i += step){
            score = chromosomeCompetition(game, bot, group[schedule[i]], group[schedule[i+step/2]]) ? 1 : -1;
            score -= chromosomeCompetition(game, bot, group[schedule[i+step/2]], group[schedule[i]]) ? 1 : -1;
            result[schedule[i]].fitness += score;
            result[schedule[i+step/2]].fitness -= score;
            if(score < 0){
                temp = schedule[i];
                schedule[i] = schedule[i+step/2];
                schedule[i+step/2] = temp;
            }
        }
        step *= 2;
    }
    sort(result, result+size, cmp);
    return result;
}

void GA::doReproduction(ChrFit *groupFit, uint32_t* group, int size, int chosen) {
    printf("doReproduction\n");
    for(int i = size-1; i >= size - chosen; i--){
        group[i] = groupFit[i].chr;
    }
}

void GA::doCrossover(ChrFit *groupFit, uint32_t *group, int size) {
    printf("doCrossover\n");
    int index;
    uint32_t chr1, chr2;

    for(int i = 0; i < size; i++){
        index = generator() % size;
        chr1 = groupFit[index].chr;
        index = generator() % size;
        chr2 = groupFit[index].chr;
        group[i] = chromosomeCrossover(chr1, chr2);
    }
    delete[] groupFit;
}

void GA::doMutation(uint32_t *group, int size) {
    printf("doMutation\n");
    for(int i = 0; i < size; i++){
        group[i] = chromosomeMutation(group[i]);
    }
}

void GA::algorithm() {
    int epoch = 1000, size = 64, chosen = 16, loss;
    uint32_t* group = generateGroup(size);

    for(int i = 0; i < epoch; i++){
        printf("Epoch:\t%d\n", i);
        ChrFit* grpFit = calculateFitness(group, size);
        loss = grpFit[size-1].fitness - grpFit[0].fitness;
        printf("Loss:\t%d\n", loss);
//        if(loss < 5){
//            break;
//        }
        doReproduction(grpFit, group, size, chosen);
        doCrossover(grpFit, group, size - chosen);
        doMutation(group, size - chosen);
        printChromosome(group[size-1]);
        printChromosome(group[0]);
    }
    destroyGroup(group);
}