from board import Board
from const import *
import numpy as np
from training.ai_v2_4train import AIV2
from ai import AI


class Game:
    def __init__(self, board, ai_1, ai_2):
        self.board = board
        self.ai_v1 = ai_1
        self.ai_v2 = ai_2
        self.step = 1

    # ---------------------------------public-----------------------------------
    def run(self, reverse):
        """
        1. 玩家就绪，AI就绪，裁判就绪
        2. 玩家选择棋子颜色
        3. 黑子先手，执棋者下棋，反转棋盘上对手棋子的颜色
        4. 裁判判断游戏是否结束，如果为否，继续
        5. 对手下棋
        6. 执行第4步
        7. 执行3~6步执导裁判判定游戏结束
        8. 裁判宣布获胜方
        """
        # 初始化
        is_finish = 0  # is_finish==2时说明双方均无棋可下
        ai_v1_color = reverse  # self.ai_v1选择的棋子颜色
        chessman_color = True  # 控制执棋方，True表示执棋颜色为黑色
        chessman_num = [0, 0]  # 棋盘上黑白棋的个数
        # 开始游戏
        while True:
            winner_color = self.check_is_finish(is_finish, chessman_num)  # 检查游戏是否结束

            self.step += 1

            if winner_color:
                break
            # 双方轮流下棋
            if ai_v1_color == chessman_color:  # 如果执棋方颜色跟最优秀的AI棋子颜色一致，轮到玩家下棋
                if self.ai_v1.think(chessman_color, self.ai_v1.name):
                    is_finish = 0
                else:
                    is_finish += 1
                chessman_color = not chessman_color  # 轮到对方下棋
            else:  # 否则轮到AI下棋
                if self.ai_v2.think(chessman_color, self.ai_v2.name):
                    is_finish = 0
                else:
                    is_finish += 1
                chessman_color = not chessman_color  # 轮到对方下棋

        # 宣布获胜方
        return self.congratulate(ai_v1_color, winner_color, chessman_num, self.ai_v1.name, self.ai_v2.name)

    def check_is_finish(self, is_finish, chessman_num):
        """
        判断游戏是否结束
        :param is_finish: is_finish==2表示双方都无棋可下，游戏结束
        :param chessman_num: 棋盘上黑白棋子的个数
        :return char or bool:False表示游戏继续，其它情况表示游戏结束。其中char用来标识哪方获胜
        """
        # 计算棋盘上棋子数
        white_num = np.count_nonzero(self.board.board == WHITE)
        black_num = np.count_nonzero(self.board.board == BLACK)
        chessman_num[0], chessman_num[1] = black_num, white_num

        # 检查棋盘是否已下满
        flag = np.any(self.board.board == BLANK)
        if not flag or is_finish >= 2:
            if black_num > white_num:  # 黑子胜
                return BLACK
            elif black_num < white_num:  # 白子胜
                return WHITE
            else:  # 平局
                return DOGFALL

        # 检查棋盘上是否只有黑棋
        flag = np.any(self.board.board == WHITE)
        if not flag:
            return BLACK

        # 检查棋盘上是否只有白棋
        flag = np.any(self.board.board == BLACK)
        if not flag:
            return WHITE

        # 上述条件都不满足，则继续下棋
        return False

    def congratulate(self, ai_v1_color, winner_color, chessman_num, ai_v1_name, ai_v2_name):
        # 输出最终棋盘
        """
        宣布获胜方
        :param ai_v1_color:玩家执棋颜色
        :param winner_color:胜利者棋子颜色。
            如果winner_color = DOGFALL，表示平局
            如果winner_color = BLACK，表示黑方胜
            如果winner_color = WHITE，表示白方胜
        :param chessman_num: 棋盘上黑白棋子的个数
        :param ai_v1_name: self.ai_v1的名称
        :param ai_v2_name: self.ai_v2的名称
        """
        self.board.xboard = self.board.board
        # 判断哪方胜利
        if winner_color == BLACK:
            if ai_v1_color:
                return 1
            else:
                return -1
        elif winner_color == WHITE:
            if not ai_v1_color:
                return 1
            else:
                return -1
        else:
            return 0

    def refresh(self):
        """
        重置棋盘
        """
        self.board.board = Board().board
        self.board.xboard = Board().board
        self.board.parity = 0
        self.step = 1
