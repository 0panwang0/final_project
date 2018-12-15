from ai_v2_4train import AIV2
from board import Board
from reversi4train import Game
import random
import copy
import json

GENE_BLOCK_SIZE = 5  # 基因的数量，即有多少个权重(一个评估函数需要一个权重)
POPULATION = 100  # 种群内个体的数量
SUPERIOR_NUM = 1  # 保留优胜种的数量，剩下的杂交
EPOCH_SAVE = 5  # 多少代保存一次最优染色体


def generate_weight():
    """
    随机产生染色体
    :return chromosome = [12, 34, 255, 222, 250] etc.
    """
    chromosome = []
    for i in range(GENE_BLOCK_SIZE):
        chromosome.append(random.randint(0, 255))
    return chromosome


def int2bit(old_chromosome):
    """
    将十进制数转为二进制数
    chromosome = [12, 34, 255, 222, 250]
    1. 使用{:08b}格式符直接将整数转为二进制字符串
        chromosome = [['00001100'], ..]
    2. 使用list将字符串切割成字符
        chromosome = [['0', '0', '0', '0', '1', '1', '0', '0'], ..]
    3. 使用map(int, list)将字符转为整数
        chromosome = [[0, 0, 0, 0, 1, 1, 0, 0], ..]
    """
    new_chromosome = []
    for part_chromosome in old_chromosome:
        new_chromosome.append(list(map(int, list('{:08b}'.format(part_chromosome)))))
    return new_chromosome


def bit2int(old_chromosome):
    """
    将二进制数转化为在[0, 255]范围内的整数
    chromosome = [[1, 0, 1, 0, 1, 1, 1, 1], [..], [..],..]
    1. 为了便于操作，使用map(str, list)先将每个基因中的每一个位变成字符
        chromosome = [['1', '0', '1', '0', '1', '1', '1', '1'], [..], [..],..]
    2. 使用str.join函数将单个基因内的8个字符拼接在一起
        chromosome = [['10101111'], [..], ..]
    3. 使用int(x, base=2)函数将字符串由2进制转为10进制。注意base=2指明x是二进制形式
        chromosome = [175, ...]
    """
    new_chromosome = list(map(lambda x: map(str, x), old_chromosome))
    return list(map(lambda x: int("".join(x), base=2), new_chromosome))


def crossover(chromosome1, chromosome2):
    """
    染色体单点交叉
    :param chromosome1: 父染色体
    :param chromosome2: 母染色体
    :return posterity_chromosome: 子染色体
    """
    node = random.randint(0, 4)
    return chromosome1[:node] + chromosome2[node:]


def mutation(chromosome):
    """
    染色体变异
    :param chromosome:
    :return:
    """
    mutation_num = int(random.random() * 10)    # 突变点的个数
    mutation_pos = random.choices([_ for _ in range(GENE_BLOCK_SIZE * 8)], k=mutation_num)  # 随机选择突变位置
    new_chromosome = copy.deepcopy(chromosome)
    for pos in mutation_pos:
        row = pos // 8  # 找到对应基因块
        col = pos % 8  # 从基因块中找到基因
        new_chromosome[row][col] ^= 1  # 取反
    return new_chromosome


def initial_population():
    """
    初始化种群
    :return group:种群，里面的每个染色体随机生成
    """
    group = []
    for i in range(POPULATION):
        group.append(generate_weight())
    return group


def count_fitness(group):
    # 每个染色体的适应度，下标与group中的一一对应
    fitness = [0] * POPULATION

    for i in range(len(group)):  # 共2 * POPULATION * (POPULATION - 1)场比赛
        for j in range(len(group)):
            if i == j:
                continue
            # 正在初始化棋盘
            board = Board()
            # 寻找染色体
            chromosome1 = group[i]
            chromosome2 = group[j]
            # 正在往AI体内插入染色体，注意归一化染色体
            ai_v1 = AIV2(board, list(map(lambda x: x / 255, chromosome1)), 3, "ai_v1")
            ai_v2 = AIV2(board, list(map(lambda x: x / 255, chromosome2)), 3, "ai_v2")
            # 双方开始比赛
            game = Game(board, ai_v1, ai_v2)
            score = game.run(True)
            # 记录比分
            fitness[i] += score
            fitness[j] -= score
            # print("第{}号染色体的适应度fitness:".format(i + 1), fitness[i])
    group_fitness = list(zip(group, fitness))  # 将染色体与它的适应度配对，[(chromosome,fitness), ...]
    return sorted(group_fitness, key=lambda x: x[1])  # 根据fitness[?][0]，也就是适应度来排序


def do_reproduction_operator(group_fitness):
    """
    选择/再生运算子
    :param group_fitness: 群落， 包括染色体和适应度，根据适应度由小到大排序。适应度越高越好
    :return:
    """
    superior = []  # 储存优胜种
    i = 0
    # 选择
    while i < SUPERIOR_NUM:
        chromosome_fitness = group_fitness.pop()  # 将最后面的pop出来，因为它的适应度最高。剩下的进行杂交。
        superior.append(chromosome_fitness[0])
        i += 1
    return superior


def do_crossover(group_fitness):
    group, fitness = zip(*group_fitness)  # 解除配对，将所有group_fitness[i][0]组成一个列表，group_fitness[i][1]同理
    new_group = []
    # -----------------------------------交叉运算-----------------------------------------
    min_v = abs(min(fitness)) + 1
    fitness = [x + min_v for x in fitness]  # 如果fitness全为0或有负数，random.choices会出错，故将所有值+|最小值|+1
    for i in range(POPULATION - SUPERIOR_NUM):
        # 将适应度频率作为权重进行加权随机选择。权重越大越容易被选到。每次选出2个染色体。
        chromosomes = random.choices(group, weights=fitness, k=2)
        new_chromosome = crossover(chromosomes[0], chromosomes[1])
        new_group.append(new_chromosome)
    # -----------------------------------交叉运算-----------------------------------------
    return new_group


def do_mutation(group, superior):
    """
    :param group: 经过交叉后的群落
    :param superior: 选择运算得到的优秀染色体
    :return: 经过突变后的群落
    """
    # 转为二进制数
    bit_group = list(map(int2bit, group))
    # 突变
    mutation_group = list(map(mutation, bit_group))
    # 转回十进制数
    int_group = list(map(bit2int, mutation_group))
    return superior + int_group


def main():
    epoch = 1
    group = initial_population()  # 群落
    while True:
        print("------------------EPOCH {}----------------------".format(epoch))
        group_fitness = count_fitness(group)  # 群落中每个个体的适应度
        loss = group_fitness[-1][1] - group_fitness[0][1]
        print("highest score:", group_fitness[-1][1], ", lowest score:", group_fitness[0][1])
        print("loss:", loss)
        if loss < 5 or epoch > 10:
            break
        superior = do_reproduction_operator(group_fitness)
        crossover_group = do_crossover(group_fitness)
        group = do_mutation(crossover_group, superior)
        epoch += 1
        if epoch % EPOCH_SAVE == 0:
            with open("save/{}.json".format(epoch), "w") as f:
                json.dump(superior[0], f)


if __name__ == '__main__':
    main()
