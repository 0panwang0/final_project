from const import *
import numpy as np
import copy


class Board:
    def __init__(self):
        self.board = np.array([
            ['┌', '─', '─', '─', '─', '─', '─', '┐'],
            ['│', '·', '·', '·', '·', '·', '·', '│'],
            ['│', '·', '·', '·', '·', '·', '·', '│'],
            ['│', '·', '·', '☺', '☻', '·', '·', '│'],
            ['│', '·', '·', '☻', '☺', '·', '·', '│'],
            ['│', '·', '·', '·', '·', '·', '·', '│'],
            ['│', '·', '·', '·', '·', '·', '·', '│'],
            ['└', '─', '─', '─', '─', '─', '─', '┘'],
        ])  # 保留未进行标记的棋盘，通过该棋盘来获得被标记的棋盘
        self.xboard = copy.deepcopy(self.board)  # 该棋盘用于标记
        self.parity = 0

    # ---------------------------------public-----------------------------------
    def find_position(self, chessman_color):
        """
        1. 扫描整个棋盘，共需要6步(因为棋盘可以落子位置有六行六列)
        2. 每一步将棋盘切成行、列、正右上角次对角线、正左下角次对角线、反右下角次对角线、反左上角次对角线共6中形式
        3. 对上述六种形式寻找可落子的位置
        :param chessman_color: bool， 执棋者的棋子颜色，True表示黑子
        :return flag:bool，True表示执棋者有位置落子
        """
        self.xboard = copy.deepcopy(self.board)
        flag = []
        for i in range(1, 7):
            row = self.xboard[i, :]  # 切出第i行
            col = self.xboard[:, i]  # 切出第j列
            flag.append(self.__forw_back_mark_pos(row, chessman_color))
            flag.append(self.__forw_back_mark_pos(col, chessman_color))
            # -花式索引获得对角线序列
            index = np.array([_ for _ in range(i + 2)])
            # --对角线(左上角到右下角)
            diag_righttop = self.xboard[index, index + 6 - i]  # 切出右上角范围中离主对角线有6-i个单位的“次对角线”
            diag_leftbottom = self.xboard[index + 6 - i, index]  # 切出左下角范围中离主对角线有6-i个单位的“次对角线”
            flag.append(self.__forw_back_mark_pos(diag_righttop, chessman_color))
            flag.append(self.__forw_back_mark_pos(diag_leftbottom, chessman_color))
            self.xboard[index, index + 6 - i] = diag_righttop
            self.xboard[index + 6 - i, index] = diag_leftbottom
            # --对角线(左下角到右上角)
            # ---切出右下角范围中离主对角线有6-i个单位的“次对角线”
            diag_rightbottom = self.xboard[(index + 6 - i), (index + 6 - i)[::-1]]
            diag_lefttop = self.xboard[index, index[::-1]]  # 切出左上角范围中离主对角线有6-i个单位的“次对角线”
            flag.append(self.__forw_back_mark_pos(diag_rightbottom, chessman_color))
            flag.append(self.__forw_back_mark_pos(diag_lefttop, chessman_color))
            self.xboard[(index + 6 - i), (index + 6 - i)[::-1]] = diag_rightbottom
            self.xboard[index, index[::-1]] = diag_lefttop
        return np.sum(np.array(flag)) > 0

    def covert_color(self, pos, chessman_color):
        """
        逆转对手棋子的颜色
        :param pos:执棋者落子的位置
        :param chessman_color:执棋者棋子的颜色
        """
        row = self.board[pos[0], :]  # 切出行
        col = self.board[:, pos[1]]  # 切出列
        self.__convert_seq(row, chessman_color)
        self.__convert_seq(col, chessman_color)
        # -花式索引获得对角线序列
        # --切出对角线(左上角到右下角)
        if pos[0] >= pos[1]:
            r_index = np.array([_ for _ in range(pos[0] - pos[1], 8)])
            c_index = np.array([_ for _ in range(0, 8 - (pos[0] - pos[1]))])
        else:
            r_index = np.array([_ for _ in range(0, 8 - (pos[1] - pos[0]))])
            c_index = np.array([_ for _ in range(pos[1] - pos[0], 8)])
        adiag = self.board[r_index, c_index]
        self.__convert_seq(adiag, chessman_color)
        self.board[r_index, c_index] = adiag
        # --切出对角线(左下角到右上角)
        if pos[0] + pos[1] < 8:
            r_index = np.array([_ for _ in range(0, pos[0] + pos[1] + 1)])
            c_index = np.array([_ for _ in range(0, pos[0] + pos[1] + 1)][::-1])
        else:
            r_index = np.array([_ for _ in range(pos[0] + pos[1] - 7, 8)])
            c_index = np.array([_ for _ in range(pos[0] + pos[1] - 7, 8)][::-1])
        diag = self.board[r_index, c_index]
        self.__convert_seq(diag, chessman_color)
        self.board[r_index, c_index] = diag
        self.board[pos[0], pos[1]] = BLACK if chessman_color else WHITE

    # ---------------------------------private-----------------------------------
    def __convert_seq(self, seq, chessman_color):
        """
        对棋盘的某个切片进行颜色反转
        :param seq:切片，注意不管是行切片、列切片还是对角线切片，得到的都是行向量或者说可以看成是行向量(numpy切片)
        :param chessman_color: bool， 执棋者的棋子颜色，True表示黑子
        """
        if chessman_color:  # 判断当前执棋者的棋子颜色
            my_color = BLACK
            oppo_color = WHITE
        else:
            my_color = WHITE
            oppo_color = BLACK
        my_index = np.argwhere(seq == CHESSMAN)  # 判断执棋者落子的位置
        if len(my_index):
            my_index = my_index[0, 0]
            i = my_index + 1
            # 从落子位置向右搜寻可翻转的棋子，再次强调seq为行向量，故有“左右”方向
            while seq[i] == oppo_color:
                i += 1
            if seq[i] == my_color:
                seq[my_index + 1:i] = my_color
            # 从落子位置向左搜寻可翻转的棋子
            i = my_index - 1
            while seq[i] == oppo_color:
                i -= 1
            if seq[i] == my_color:
                seq[i + 1:my_index] = my_color

    def __mark_position(self, sequence, chessman_color):
        """
        检查给定的序列中空白的位置是否可以落子并标记
        :param sequence:给定的序列，其中可能包括空位置、黑子、白字
        :param chessman_color: bool， 执棋者的棋子颜色，True表示黑子
        :return flag:bool，True表示执棋者有位置落子
        """
        if chessman_color:  # 判断当前执棋者的棋子颜色
            my_color = BLACK
            oppo_color = WHITE
            chessman = BLACK_TAB
        else:
            my_color = WHITE
            oppo_color = BLACK
            chessman = WHITE_TAB
        j = 1
        flag = False  # 判断当前序列是否有位置落子
        while j < sequence.shape[0]:  # 检查并标记
            if sequence[j] == oppo_color:  # 首先找到对手的棋子，比如在位置a上
                k = j + 1
                while sequence[k] == oppo_color:  # 从a开始往右到位置b都是对手的棋子
                    k += 1
                # [...a-1, (a, a+1, ... b), b+1]
                # 如果位置b+1是执棋者的棋子，并且位置a-1是空位，显然在a-1落子可以翻转对手的棋子
                if sequence[k] == my_color and sequence[j - 1] == BLANK:
                    sequence[j - 1] = chessman
                    flag = True
                j = k
            j += 1
        return flag

    def __forw_back_mark_pos(self, sequence, chessman_color):
        """
        因为__mark_position函数只能找出单边的可以下棋的位置，
        故需要调用__mark_position函数两次，分别从左边和右边分别找出可以下棋的位置
        :param sequence:序列(棋盘的一个切片)
        :param chessman_color: bool， 执棋者的棋子颜色，True表示黑子
        :return flag:bool，True表示执棋者有位置落子
        """
        flag = False  # 判断执棋者是否有位置落子
        ok = self.__mark_position(sequence, chessman_color)  # 从左查找空白位是否可以落子
        seq_reverse = sequence[::-1]  # 倒叙，表示从右边开始搜寻可以落子的位置
        ok1 = self.__mark_position(seq_reverse, chessman_color)  # 从右查找空白位是否可以落子
        if ok or ok1:
            flag = True
        return flag
