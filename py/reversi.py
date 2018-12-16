import gym
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ReversiEnv(gym.Env):
    # --------------------------------------global-----------------------------------------
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    BLACK = -1
    WHITE = 1
    DRAW = 65535
    GAMING = 65534
    BOARD_WIDTH = 8
    BOARD_SIZE = 64
    MASK = 0xffff_ffff_ffff_ffff

    def __init__(self):
        self.black_board = 0
        self.white_board = 0
        self.skip_count = 0

        '''棋盘中央放入四个棋子，黑白棋子各两个'''
        self.black_board |= (1 << self.__get_index(3, 4)) | (1 << self.__get_index(4, 3))
        self.white_board |= (1 << self.__get_index(3, 3)) | (1 << self.__get_index(4, 4))

        self.directions = [self.__to_n, self.__to_s, self.__to_w, self.__to_e, self.__to_nw, self.__to_ne, self.__to_sw,
                           self.__to_se]

        self.viewer = None

    # --------------------------------------public-----------------------------------------
    def get_valid_pos(self, my_color):
        """
        根据棋子颜色获得该棋子所能下棋的位置
        :param my_color: 我的棋子的颜色
        :return valid_pos: 能够落子的位置
        """
        pos = 0
        emp = self.__get_empty()

        if my_color == self.BLACK:
            my = self.black_board
            opp = self.white_board
        else:
            my = self.white_board
            opp = self.black_board

        for i in range(8):
            tmp = self.directions[i](my) & opp & self.MASK
            for j in range(5):
                tmp |= self.directions[i](tmp) & opp
            pos |= self.directions[i](tmp) & emp

        return self.board_to_list(pos)

    def flip(self, action, color):
        if color == self.BLACK:
            my = self.black_board
            opp = self.white_board
        elif color == self.WHITE:
            my = self.white_board
            opp = self.black_board

        pos = 1 << action
        for i in range(8):
            tmp = self.directions[i](pos)
            mask = 0
            while tmp & opp:
                mask |= tmp
                tmp = self.directions[i](tmp)

            if tmp & my:
                my = (my ^ mask) & self.MASK
                opp = (opp ^ mask) & self.MASK

        if color == self.BLACK:
            self.black_board = my
            self.white_board = opp
        elif color == self.WHITE:
            self.white_board = my
            self.black_board = opp

    def is_over(self):
        '''
        结束游戏有两种情况：
        1. 棋盘上没有空位；
        2. 棋盘上有空位，但是双方都无处落子。
        '''
        if self.skip_count == 2:
            return True
        if self.__get_empty() == 0:
            return True
        return False

    def winner(self):
        if self.is_over():
            black_piece = self.__count_bits(self.black_board)
            white_piece = self.__count_bits(self.white_board)
            if black_piece > white_piece:
                return self.BLACK
            elif black_piece == white_piece:
                return self.DRAW
            else:
                return self.WHITE
        else:
            return self.GAMING

    def skip(self):
        self.skip_count += 1

    def clear_skip(self):
        self.skip_count = 0

    def step(self, action):
        """
        ###修改父类中的step函数，请不要修改函数名###
        该函数用于翻转对方棋子
        :param action: 包括坐标、执棋方棋子颜色、我的棋子颜色  例如：[55,1, 1] 表示： 坐标(55 // 8, 55 % 8), 白棋, 白棋
        :return:下一个状态，动作价值，是否结束
        """
        win_reward = 1000
        lose_reward = -1000
        draw_reward = 0
        gaming_reward = 0

        if action:
            self.clear_skip()
            self.flip(action[0], action[1])
            board = self.__get_board()
            winner = self.winner()
            if winner == action[2]:
                return board, win_reward, winner
            elif winner == self.DRAW:
                return board, draw_reward, winner
            elif winner == self.GAMING:
                return board, gaming_reward, winner
            else:  # 对方胜利
                return board, lose_reward, winner

        else:
            self.skip()

    def reset(self):
        self.black_board = 0
        self.white_board = 0

        '''棋盘中央放入四个棋子，黑白棋子各两个'''
        self.black_board |= (1 << self.__get_index(3, 4)) | (1 << self.__get_index(4, 3))
        self.white_board |= (1 << self.__get_index(3, 3)) | (1 << self.__get_index(4, 4))

    def board_to_list(self, board):
        """
        64位的board转换到list，
        list的元素是从小到大排列的
        """
        res = []
        while board:
            res.append(self.__low_bit_index(board))
            board -= self.__low_bit(board)

        return res

    # --------------------------------------private-----------------------------------------
    def __get_empty(self):
        return (~ (self.black_board | self.white_board)) & self.MASK

    def __to_n(self, x):
        return (x >> 8) & self.MASK

    def __to_s(self, x):
        return (x << 8) & self.MASK

    def __to_w(self, x):
        return ((x & 0xfefefefefefefefe) >> 1) & self.MASK

    def __to_e(self, x):
        return ((x & 0x7f7f7f7f7f7f7f7f) << 1) & self.MASK

    def __to_nw(self, x):
        return self.__to_n(self.__to_w(x))

    def __to_ne(self, x):
        return self.__to_n(self.__to_e(x))

    def __to_sw(self, x):
        return self.__to_s(self.__to_w(x))

    def __to_se(self, x):
        return self.__to_s(self.__to_e(x))

    def __get_index(self, x, y):
        return x * self.BOARD_WIDTH + y

    def __low_bit(self, x):
        return x & (-x)

    '''
    gmpy库有高效的实现，但是跟新版本python不兼容
    bit_length性能未知
    可能需要改进算法
    '''

    def __low_bit_index(self, x):
        return self.__low_bit(x).bit_length() - 1

    '''
    https://stackoverflow.com/questions/9829578/fast-way-of-counting-non-zero-bits-in-positive-integer
    '''

    def __count_bits(self, x):
        x = (x & 0x5555555555555555) + ((x & 0xaaaaaaaaaaaaaaaa) >> 1)
        x = (x & 0x3333333333333333) + ((x & 0xcccccccccccccccc) >> 2)
        x = (x & 0x0f0f0f0f0f0f0f0f) + ((x & 0xf0f0f0f0f0f0f0f0) >> 4)
        x = (x & 0x00ff00ff00ff00ff) + ((x & 0xff00ff00ff00ff00) >> 8)
        x = (x & 0x0000ffff0000ffff) + ((x & 0xffff0000ffff0000) >> 16)
        x = (x & 0x00000000ffffffff) + ((x & 0xffffffff00000000) >> 32)
        return x

    def __get_board(self):
        black_list = np.array(self.board_to_list(self.black_board))
        white_list = np.array(self.board_to_list(self.white_board))
        board = np.zeros([self.BOARD_SIZE])
        board[black_list] = self.BLACK
        board[white_list] = self.WHITE
        return board


if __name__ == '__main__':
    env = ReversiEnv()
    print(env.get_valid_pos(-1))
    # env.flip(20, 1)
