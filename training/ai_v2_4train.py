import numpy as np
from const import *
import copy


class AIV2:
    def __init__(self, board, weights=(0.5, 0.5, 0.5, 0.5, 0.5), depth=4, name='ai'):
        self.board = board  # 棋盘
        self.pos_weight = np.array([
            ['┌', '─', '─', '─', '─', '─', '─', '┐'],
            ['│', 20, 5, 5, 5, 5, 20, '│'],
            ['│', 5, -10, -3, -3, -10, 5, '│'],
            ['│', 5, -3, 3, 3, -3, 5, '│'],
            ['│', 5, -3, 3, 3, -3, 5, '│'],
            ['│', 5, -10, -3, -3, -10, 5, '│'],
            ['│', 20, 5, 5, 5, 5, 20, '│'],
            ['└', '─', '─', '─', '─', '─', '─', '┘']
        ])  # 棋盘权重表，为了跟棋盘对应，这里加上了边框
        self.name = name
        self.weights = weights
        self.depth = depth

    # ---------------------------------public-----------------------------------
    def think(self, chessman_color, oppo_name):
        """
        AI假装在思考
        :param chessman_color:AI棋子颜色
        :return bool:False表示无位置可下
        :param oppo_name: 对手AI的名字
        """
        ok = self.board.find_position(chessman_color)  # 判断自己是否有位置可以落子
        if ok:  # 有位置下棋
            # alpha_beta剪枝
            _, pos = self.alpha_beta_minimax(chessman_color, self.depth, parity=self.board.parity)

            # AI下棋
            self.__chess(pos, chessman_color)
        else:   # 不能下棋，奇偶性发生变化
            self.board.parity += 1
        return True if ok else False

    def alpha_beta_minimax(self, chessman_color, depth=4, parity=0, my_score=-float('inf'), oppo_score=float('inf')):
        """
        alpha_beta剪枝版DFS递归minmax
        :param chessman_color:执棋者棋子颜色
        :param depth:向“未来”看depth步
        :param parity: 奇偶性
        :param my_score:储存自己的得分，初始化为负无穷。因为需要进行结点间得分的比较(所以必须初始化为负无穷)，自己的得分越高越好。
        :param oppo_score:初始化为正无穷，因为对手是min结点，它的值越小越好
        :return score, pos:返回得分最高的落子位置
        """
        best_pos = None
        best_score = my_score
        valid_pos = self.__get_valid_pos(chessman_color)

        # 最多向前看depth步，如果depth=0说明已经看到了"未来的"情况了，回溯
        if depth == 0:  # 到达伪叶子结点，计算得分，回溯
            return self.evaluate(chessman_color, valid_pos, parity), best_pos

        # 处于中间结点，继续下棋
        if len(valid_pos) == 0:  # 如果无位置可落子，轮到对方下棋
            # 备份，供下面回溯用
            board = self.board.board
            xboard = self.board.xboard
            self.board.board = copy.deepcopy(self.board.board)
            self.board.xboard = copy.deepcopy(self.board.xboard)
            # 使用minmax，模拟对手，此时到对方回合
            self.board.find_position(not chessman_color)  # 对方正在判断是否有位置可以落子
            score, _ = self.alpha_beta_minimax(not chessman_color, depth - 1, parity + 1, -oppo_score,
                                               -best_score)  # 对方也使用minmax策略
            # 使用了递归，需要回溯
            self.board.board = board
            self.board.xboard = xboard
            if -score > best_score:
                best_score = -score

        # 遍历所有可落子的位置，找到得分最高的情况
        for pos in valid_pos:
            # 备份，供下面回溯用
            board = self.board.board
            xboard = self.board.xboard
            self.board.board = copy.deepcopy(self.board.board)
            self.board.xboard = copy.deepcopy(self.board.xboard)
            # 下棋
            self.__chess(pos, chessman_color)
            # 使用minmax，模拟对手，此时到对方回合
            self.board.find_position(not chessman_color)  # 对方正在判断是否有位置可以落子
            score, _ = self.alpha_beta_minimax(not chessman_color, depth - 1, parity + 1, -oppo_score,
                                               -best_score)  # 对方也使用minmax策略
            # 使用了递归，需要回溯
            self.board.board = board
            self.board.xboard = xboard
            # minimax
            if -score > best_score:
                best_score = -score
                best_pos = pos
            """
                剪枝。注意oppo_score是祖宗节点中对手(min)的得分。刚开始初始化为无穷。假设我们已经探索了一些结点
            使得oppo_score有值。这时，如果自己的得分best_score大于对手的得分(oppo_score)，显然对手不会采纳自己
            的得分。这由min结点的性质可知。
                而自己肯定希望得分越高越好，所以如果自己继续探索下一层的后续结点，并从中获得得分score1。只有当
            score>best_score时自己才会采纳该score1。而对手已经表明不会采纳自己的best_score，所以更不可能采纳
            score1。因此探索后续子节点做了无用功。剪枝。
            """
            if best_score > oppo_score:
                break
        return best_score, best_pos

    def evaluate(self, chessman_color, valid_pos, parity):
        """
        多种估计函数的综合
        :param chessman_color:
        :param valid_pos:
        :param parity:
        :return:
        """
        return \
            self.weights[0] * self.__eval_parity(chessman_color, parity) + \
            self.weights[1] * self.__eval_stable_p(chessman_color) + \
            self.weights[2] * self.__eval_movers(valid_pos) + \
            self.weights[3] * self.__eval_chessman_num(chessman_color) + \
            self.weights[4] * self.__eval_overall_pos(chessman_color)

    # ---------------------------------private-----------------------------------
    def __get_valid_pos(self, chessman_color):
        """
        获取可落子的位置
        :param chessman_color: 执棋者棋子颜色
        :return pos:可落子位置
        """
        tab = BLACK_TAB if chessman_color else WHITE_TAB
        valid_pos = np.argwhere(self.board.xboard == tab)
        return valid_pos

    def __chess(self, pos, chessman_color):
        """
        执棋者下棋
        :param pos:落子位置
        :param chessman_color:执棋者棋子颜色
        """
        self.board.board[pos[0], pos[1]] = CHESSMAN  # 暂时用'O'填充到落子所在的位置，方便接下来将对手棋子进行反转
        self.board.covert_color(pos, chessman_color)  # 反转对方棋子颜色

    # --------------------------------估计函数----------------------------------------
    def __eval_parity(self, chessman_color, parity):
        """
            判断奇偶性，parity初始为0。如果整局游戏双方都没有出现一方不能落子的情况，那么每一轮场上多两个棋子。
        对于6*6棋盘来说，共有偶数个位置。因此最后一轮必定是黑方先落子，然后白方落子。这样在最后一轮中白方
        能反转黑方棋子，但黑方不能反转白方棋子。因此白方占优。
            但是，如果游戏过程中有一方不能落子，那么最后一轮会变成黑方最后落子，此时黑方占优。进一步地，如果游戏中
        出现多次一方不能落子的情况，那么最后一次落子方也会发生变化。因此用parity统计发生了多少次不能落子的情况。
            显然对于黑方来说parity因该为奇数。对于白方来说parity应该为偶数
        :param chessman_color:执棋者棋子颜色
        :param parity:奇偶性
        :return score:
        """
        my_color = BLACK if chessman_color else WHITE  # 自己棋子的颜色
        if parity % 2 == my_color:
            return parity * 2 if parity % 2 == 0 else (parity + 1) * 2
        else:
            return 0

    def __eval_stable_p(self, chessman_color):
        """
        计算棋子稳定点个数。为了提高速率，实际上只计算四条边上稳定点的个数
        :param chessman_color:执棋者棋子颜色
        :return stable_points:稳定点数
        """
        my_color = BLACK if chessman_color else WHITE  # 自己棋子的颜色
        up_row = self.board.board[1, :]
        down_row = self.board.board[-2, :]
        left_col = self.board.board[:, 2]
        right_col = self.board.board[:, -2]

        stable_points = 0
        # 模糊计算，当边界都填满棋子时这些棋子肯定不能再被反转。本算法只考虑这种情况的稳定点
        # 这里计算时先避开四个端点，稍后再计算
        if not np.any(up_row == BLANK):
            stable_points += np.sum((up_row == my_color)[2:-2])
        if not np.any(down_row == BLANK):
            stable_points += np.sum((down_row == my_color)[2:-2])
        if not np.any(left_col == BLANK):
            stable_points += np.sum((left_col == my_color)[2:-2])
        if not np.any(right_col == BLANK):
            stable_points += np.sum((right_col == my_color)[2:-2])
        # 四个端点
        stable_points += (self.board.board[1, 1] == my_color)
        stable_points += (self.board.board[1, 6] == my_color)
        stable_points += (self.board.board[6, 1] == my_color)
        stable_points += (self.board.board[6, 6] == my_color)
        return stable_points

    def __eval_movers(self, valid_pos):
        """
        行动力
        :param valid_pos: 可落子的位置
        """
        return len(valid_pos)

    def __eval_chessman_num(self, chessman_color):
        """
        棋子数
        :param chessman_color:执棋者棋子颜色
        """
        my_color = BLACK if chessman_color else WHITE  # 自己棋子的颜色
        white_num = np.count_nonzero(self.board.board == WHITE)
        black_num = np.count_nonzero(self.board.board == BLACK)
        return black_num - white_num if my_color else white_num - black_num

    def __eval_overall_pos(self, chessman_color):
        """
        评估伪叶子节点的得分。
        1. 根据权重表计算自己的得分
        2. 根据权重表计算对手的得分
        3， score = 1式 - 2式
        :param chessman_color: 执棋者棋子颜色
        :return score: 得分
        """
        score = 0  # 得分
        my_color = BLACK if chessman_color else WHITE  # 自己棋子的颜色
        oppo_color = WHITE if chessman_color else BLACK  # 对手棋子的颜色

        # 获得双方棋子的位置
        my_pos = np.argwhere(self.board.board == my_color)
        oppo_pos = np.argwhere(self.board.board == oppo_color)

        # 计算得分
        for p in my_pos:
            score += int(self.pos_weight[p[0], p[1]])
        for p in oppo_pos:
            score -= int(self.pos_weight[p[0], p[1]])

        return score
    # --------------------------------估计函数----------------------------------------
