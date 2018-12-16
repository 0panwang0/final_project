import gym
import logging

logger = logging.getLogger(__name__)


class ReversiEnv(gym.Env):
    # --------------------------------------global-----------------------------------------
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    BLACK = -1
    WHITE = 1
    BOARD_WIDTH = 8
    BOARD_SIZE = 64
    MASK = 0xffff_ffff_ffff_ffff

    def __init__(self):
        self.black_board = 0
        self.white_board = 0

        '''棋盘中央放入四个棋子，黑白棋子各两个'''
        self.black_board |= (1 << self.__get_index(3, 4)) | (1 << self.__get_index(4, 3))
        self.white_board |= (1 << self.__get_index(3, 3)) | (1 << self.__get_index(4, 4))

        self.directions = [self.__to_n, self.__to_s, self.__to_w, self.__to_e, self.__to_nw, self.__to_ne, self.__to_sw, self.__to_se]

        self.viewer = None

        self.count = 0  # 计数器

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
            tmp = self.directions[i](my) & opp
            for j in range(5):
                tmp |= self.directions[i](tmp) & opp
            pos |= self.directions[i](tmp) & emp

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
        self.black_board = 0
        self.white_board = 0

        '''棋盘中央放入四个棋子，黑白棋子各两个'''
        self.black_board |= (1 << self.__get_index(3, 4)) | (1 << self.__get_index(4, 3))
        self.white_board |= (1 << self.__get_index(3, 3)) | (1 << self.__get_index(4, 4))

        self.count = 0
        # return self.board

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
        self.__low_bit(x).bit_length()

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
