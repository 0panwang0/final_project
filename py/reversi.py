import gym
import logging
import numpy
import random
from gym import spaces

logger = logging.getLogger(__name__)

BLACK = -1
WHITE = 1
BOARD_WIDTH = 8
BOARD_SIZE = 64
MASK = 0xffff_ffff_ffff_ffff

def to_n(x):
    return (x >> 8) & MASK

def to_s(x):
    return (x << 8) & MASK

def to_w(x):
    return ((x & 0xfefefefefefefefe) >> 1) & MASK

def to_e(x):
    return ((x & 0x7f7f7f7f7f7f7f7f) << 1) & MASK

def to_nw(x):
    return to_n(to_w(x))

def to_ne(x):
    return to_n(to_e(x))

def to_sw(x):
    return to_s(to_w(x))

def to_se(x):
    return to_s(to_e(x))

def get_index(x, y):
    return x * BOARD_WIDTH + y

def low_bit(x):
    return x & (-x)

'''
gmpy库有高效的实现，但是跟新版本python不兼容
bit_length性能未知
可能需要改进算法
'''
def low_bit_index(x):
    low_bit(x).bit_length()

'''
https://stackoverflow.com/questions/9829578/fast-way-of-counting-non-zero-bits-in-positive-integer
'''
def count_bits(x):
    x = (x & 0x5555555555555555) + ((x & 0xaaaaaaaaaaaaaaaa) >> 1)
    x = (x & 0x3333333333333333) + ((x & 0xcccccccccccccccc) >> 2)
    x = (x & 0x0f0f0f0f0f0f0f0f) + ((x & 0xf0f0f0f0f0f0f0f0) >> 4)
    x = (x & 0x00ff00ff00ff00ff) + ((x & 0xff00ff00ff00ff00) >> 8)
    x = (x & 0x0000ffff0000ffff) + ((x & 0xffff0000ffff0000) >> 16)
    x = (x & 0x00000000ffffffff) + ((x & 0xffffffff00000000) >> 32)
    return x


class ReversiEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.black_board = 0
        self.white_board = 0

        '''棋盘中央放入四个棋子，黑白棋子各两个'''
        self.black_board |= (1 << get_index(3, 4)) | (1 << get_index(4, 3))
        self.white_board |= (1 << get_index(3, 3)) | (1 << get_index(4, 4))

        self.directions = [to_n, to_s, to_w, to_e, to_nw, to_ne, to_sw, to_se]

        self.viewer = None

        self.count = 0  # 计数器

    def get_valid_pos(self, my_color):
        """
        根据棋子颜色获得该棋子所能下棋的位置
        :param my_color: 我的棋子的颜色
        :return valid_pos: 能够落子的位置
        """
        pos = 0
        tmp = 0
        emp = self.get_empty()

        if my_color == BLACK:
            my = self.black_board
            opp = self.white_board
        else:
            my = self.white_board
            opp = self.black_board

        for i in range(8):
            tmp = self.directions[i](my) & opp
            for j in range(5):
                tmp |= self.directions[i](tmp) & tmp
            pos |= self.directions[i](tmp) & tmp

        return self.board_to_list(pos)
        

    def step(self, action):
        """
        ###修改父类中的step函数，请不要修改函数名###
        该函数用于翻转对方棋子
        :param action: 包括坐标和棋子颜色  例如：[1,3,1] 表示： 坐标(1,3)，白棋
        :return:下一个状态，动作价值，是否结束
        """
        pass

    def reset(self):
        self.board = [[0 for _ in range(self.SIZE)] for _ in range(self.SIZE)]

        # 棋盘中央放入四个棋子，黑白棋子各两个
        """写入代码"""
        pass

        self.count = 0
        return self.board

    def get_empty(self):
        return (~ (self.black_board | self.white_board)) & MASK


    @staticmethod
    def board_to_list(board):
        '''
        64位的board转换到list，
        list的元素是从小到大排列的
        '''
        res = []
        while board:
            res.append(low_bit_index(board))
            board -= low_bit(board)

        return res