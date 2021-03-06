import tensorflow as tf
import numpy as np
import random
from collections import deque
import time
from reversi import ReversiEnv

GAMMA = 0.9  # targetQ保留率
INITIAL_EPSILON = 0.1  # 起始随机游走概率
FINAL_EPSILON = 0.01  # 最终随机游走概率，也就是说概率不能比这个还低
OBSERVE = 500  # 先观察OBSERVE次，然后再训练
REPLAY_MEMORY = 10000  # 经验回放缓存大小
BATCH_SIZE = 200  # 每一批的训练量
SYNCHRONOUS = 100  # 目标网络同步的训练次数
VALIDATE = 10000  # 每VALIDATE次查看一次训练效果

EPISODE = 10000  # 比赛次数
SAVE_EPISODE = 200  # 每比赛SAVE_EPOCH次保存一次模型


class DQN:
    def __init__(self, env):
        self.env = env  # 黑白棋环境
        self.buffer = deque(maxlen=REPLAY_MEMORY)  # 经验池
        self.opp_buffer = deque(maxlen=REPLAY_MEMORY)
        self.epsilon = INITIAL_EPSILON  # 随机游走率
        self.hide_layer_nums = 256  # 隐藏层数量
        self.sess = None  # 会话

        self.MY_COLOR = self.env.BLACK  # 我的棋子的颜色
        self.OPP_COLOR = self.env.WHITE  # 对手棋子的颜色

        # 输入层
        self.board_input = tf.placeholder("float", [None, self.env.BOARD_SIZE])

        # Q网络
        self.b_Qtable, self.b_Q_Weights, self.w_Qtable, self.w_Q_Weights = self.__createNetwork()
        self.b_TargetQtable, self.b_TargetQ_Weights, self.w_TargetQtable, self.w_TargetQ_Weights = self.__createNetwork()

        # 定义优化器
        self.action_input, self.y_input, self.b_optimizer, self.b_loss, self.w_optimizer, self.w_loss = self.__buildOptimizer()

    # ------------------------------------public------------------------------------
    def run(self):
        # ----------------------------initial----------------------------
        cnt = 1  # 计数器，储存训练次数

        saver = tf.train.Saver()  # 储存器

        with tf.Session() as sess:
            self.sess = sess

            checkpoint = tf.train.latest_checkpoint('save/')
            if checkpoint:
                saver.restore(self.sess, checkpoint)
                print("加载之前的模型")
            else:
                print("未发现之前的模型，重新开始训练")
                self.sess.run(tf.initialize_all_variables())
                self.copyWeightsToTarget()

            # ----------------------------initial----------------------------
            for episode in range(EPISODE):
                print('--------------EPISODE:', episode, '--------------')

                board = self.env.reset()  # 获得棋盘
                # color = self.MY_COLOR  # 执棋方的颜色，初始为我的颜色
                # 开始训练
                turn = 0  # 控制执棋方
                while True:
                    action = self.epsilon_greedy(board, self.MY_COLOR)  # 获取下一步的行动

                    new_board, reward, terminal = self.env.step(
                        (int(action), self.MY_COLOR, self.MY_COLOR))  # 根据行动得到下一状态、奖励、游戏是否终止三个参数

                    if action != -1:
                        action_list = np.zeros(self.env.BOARD_SIZE)
                        action_list[action] = 1
                        # 存放到经验池
                        self.buffer.append([board, action_list, reward, new_board, terminal])
                        if terminal != self.env.GAMING:  # 结束比赛
                            self.opp_buffer.append([board, action_list, -reward, new_board, terminal])

                        # 当经验池中的数据足够多时开始训练
                        if len(self.buffer) > OBSERVE:
                            cnt += 1
                            loss = self.trainNetwork(True)
                            print("b_loss:", loss)

                            # 每训练SYNCHRONOUS次同步一次目标网络
                            if cnt % SYNCHRONOUS == 0:
                                self.copyWeightsToTarget()

                    board = new_board  # 更新棋盘
                    if terminal != self.env.GAMING:  # 结束比赛
                        winner = "黑方" if terminal == self.env.BLACK else "白方" if terminal == self.env.WHITE else "平局"
                        print('胜利方为' + winner)
                        break
                    # --------------------------------------------------------------------
                    action = self.epsilon_greedy(board, self.OPP_COLOR)  # 获取下一步的行动

                    new_board, reward, terminal = self.env.step(
                        (int(action), self.OPP_COLOR, self.OPP_COLOR))  # 根据行动得到下一状态、奖励、游戏是否终止三个参数

                    if action != -1:
                        action_list = np.zeros(self.env.BOARD_SIZE)
                        action_list[action] = 1
                        # 存放到经验池
                        self.opp_buffer.append([board, action_list, reward, new_board, terminal])
                        if terminal != self.env.GAMING:  # 结束比赛
                            self.buffer.append([board, action_list, -reward, new_board, terminal])
                        # 当经验池中的数据足够多时开始训练
                        if len(self.opp_buffer) > OBSERVE:
                            cnt += 1
                            loss = self.trainNetwork(False)
                            print("w_loss:", loss)

                            # 每训练SYNCHRONOUS次同步一次目标网络
                            if cnt % SYNCHRONOUS == 0:
                                self.copyWeightsToTarget()

                    if terminal != self.env.GAMING:  # 结束比赛
                        winner = "黑方" if terminal == self.env.BLACK else "白方" if terminal == self.env.WHITE else "平局"
                        print('胜利方为' + winner)
                        break
                    turn += 1

                # 随机游走率随着迭代次数逐渐降低
                self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EPISODE

                # 每SAVE_EPISODE次保存一次模型
                if (episode + 1) % SAVE_EPISODE == 0:
                    saver.save(self.sess, 'save/', global_step=episode)

                # 每VALIDATE次测试一次效果
                if (episode + 1) % VALIDATE == 0:
                    self.validate()

    def validate(self):
        """
        用于测试训练效果
        """
        board = self.env.reset()
        color = self.MY_COLOR  # 执棋方的颜色，初始为我的颜色
        turn = 0  # 控制执棋方

        while True:
            print("------------------------step {}--------------------------".format(turn + 1))
            self.env.render()
            action = self.action(board, color)
            _, _, terminal = self.env.step((int(action), color, self.MY_COLOR))

            # 双方轮流下棋
            color = self.OPP_COLOR if turn % 2 == 0 else self.MY_COLOR

            time.sleep(1)
            turn += 1
            if terminal != self.env.GAMING:  # 结束比赛
                print("------------------------step {}--------------------------".format(turn + 1))
                self.env.render()
                winner = "黑方" if terminal == self.env.BLACK else "白方" if terminal == self.env.WHITE else "平局"
                print('胜利方为' + winner)
                break

    def Game(self, ai_color):
        """
        人机交互游戏
        :param ai_color: AI棋子颜色
        """
        saver = tf.train.Saver()  # 储存器
        with tf.Session() as sess:
            self.sess = sess

            checkpoint = tf.train.latest_checkpoint('save/')
            if checkpoint:
                saver.restore(self.sess, checkpoint)
                print("加载之前的模型")
            else:
                print("错误！未发现之前的模型")
                return False

            # 开始游戏
            human_turn = 0 if ai_color == self.env.WHITE else 1

            board = self.env.reset()
            color = self.env.BLACK  # 执棋方的颜色，初始为黑色
            turn = 0  # 控制执棋方
            terminal = self.env.GAMING  # 设置游戏状态为正在玩游戏

            print("------------------------step {}--------------------------".format(0))
            self.env.render()
            while True:
                print("------------------------step {}--------------------------".format(turn + 1))
                valid_pos = self.env.get_valid_pos(color)
                if len(valid_pos):
                    if turn % 2 == human_turn:
                        action = -1
                        while action not in valid_pos:
                            try:
                                pos = input("请输入落子位置：")
                                pos = list(map(int, pos.split(' ')))
                                action = pos[0] * self.env.BOARD_WIDTH + pos[1]
                                if action not in valid_pos:
                                    print("位置错误")
                            except:
                                print("输入格式错误")
                        _, _, terminal = self.env.step((action, color, ai_color))

                    else:
                        action = self.action(board, color)
                        _, _, terminal = self.env.step((int(action), color, ai_color))
                        print("AI落子位置为：", action // self.env.BOARD_WIDTH, action % self.env.BOARD_WIDTH)
                else:
                    print("无子可下。")
                # 双方轮流下棋
                color = self.env.WHITE if color == self.env.BLACK else self.env.BLACK

                turn += 1
                if terminal != self.env.GAMING:  # 结束比赛
                    print("------------------------step {}--------------------------".format(turn + 1))
                    self.env.render()
                    winner = "黑方" if terminal == self.env.BLACK else "白方" if terminal == self.env.WHITE else "平局"
                    print('胜利方为' + winner)
                    return True
                self.env.render()

    def copyWeightsToTarget(self):
        """
        targetQ <- Q，详见2015年的那篇论文
        """
        for i in range(len(self.b_Q_Weights)):
            self.sess.run(tf.assign(self.b_TargetQ_Weights[i], self.b_Q_Weights[i]))
            self.sess.run(tf.assign(self.w_TargetQ_Weights[i], self.w_Q_Weights[i]))

    def trainNetwork(self, flag):
        if flag:
            minibatch = random.sample(self.buffer, BATCH_SIZE)
            TargetQtable = self.b_TargetQtable
            optimizer = self.b_optimizer
            my_loss = self.b_loss
        else:
            minibatch = random.sample(self.opp_buffer, BATCH_SIZE)
            TargetQtable = self.w_TargetQtable
            optimizer = self.w_optimizer
            my_loss = self.w_loss

        # board, action, reward, next_board
        b_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        n_b_batch = [d[3] for d in minibatch]

        y_batch = []
        Qtable_batch = TargetQtable.eval(feed_dict={self.board_input: n_b_batch})
        for i in range(BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal == self.env.GAMING:
                y_batch.append(r_batch[i] + GAMMA * np.max(Qtable_batch[i]))
            else:
                y_batch.append(r_batch[i])
        _, loss = self.sess.run([optimizer, my_loss], feed_dict={
            self.y_input: y_batch,
            self.action_input: a_batch,
            self.board_input: b_batch
        })
        return loss

    def epsilon_greedy(self, board, my_color):
        """
        这个是用于训练的
        获取落子位置
        """
        if my_color == self.env.BLACK:
            my_Qtable = self.b_Qtable
        else:
            my_Qtable = self.w_Qtable

        Qtable = my_Qtable.eval(feed_dict={
            self.board_input: [board]
        })[0]  # 输入的是一个board，输出为列表的列表，取第一个元素，它对应的就是board产生的影响

        # 获取合法的落子位置
        valid_action = self.env.get_valid_pos(my_color)
        if not valid_action:
            return -1

        for i in range(len(Qtable)):
            # 将不合法的位置权重改为负无穷
            if i not in valid_action:
                Qtable[i] = -float('inf')

        if random.random() <= self.epsilon:  # 随机游走
            return random.choice(valid_action)
        else:  # 取权重最大的那个落子位置
            return np.argmax(Qtable)

    def action(self, board, my_color):
        """
        这个是用于检验效果的，所以不用随机游走
        获取落子位置
        """

        Qtable = self.b_Qtable.eval(
            feed_dict={
                self.board_input: [board]
            })[0]  # 输入的是一个board，输出为列表的列表，取第一个元素，它对应的就是board产生的影响

        # 获取合法的落子位置
        valid_action = self.env.get_valid_pos(my_color)
        for i in range(len(Qtable)):
            # 将不合法的位置权重改为负无穷
            if i not in valid_action:
                Qtable[i] = -float('inf')

        return np.argmax(Qtable)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    # ------------------------------------private------------------------------------
    def __createNetwork(self):
        # 输入层权重
        b_W1 = self.weight_variable([self.env.BOARD_SIZE, self.hide_layer_nums])
        b_b1 = self.bias_variable([self.hide_layer_nums])
        # 隐藏层1权重
        b_W2 = self.weight_variable([self.hide_layer_nums, self.env.BOARD_SIZE])
        b_b2 = self.bias_variable([self.env.BOARD_SIZE])
        # 定义隐藏层
        b_h_layer1 = tf.nn.sigmoid(tf.matmul(self.board_input, b_W1) + b_b1)
        # 定义table
        b_table = tf.matmul(b_h_layer1, b_W2) + b_b2
        # 保存权重
        b_weights = [b_W1, b_b1, b_W2, b_b2]

        # 输入层权重
        w_W1 = self.weight_variable([self.env.BOARD_SIZE, self.hide_layer_nums])
        w_b1 = self.bias_variable([self.hide_layer_nums])
        # 隐藏层1权重
        w_W2 = self.weight_variable([self.hide_layer_nums, self.env.BOARD_SIZE])
        w_b2 = self.bias_variable([self.env.BOARD_SIZE])
        # 定义隐藏层
        w_h_layer1 = tf.nn.sigmoid(tf.matmul(self.board_input, w_W1) + w_b1)
        # 定义table
        w_table = tf.matmul(w_h_layer1, w_W2) + w_b2
        # 保存权重
        w_weights = [w_W1, w_b1, w_W2, w_b2]
        return b_table, b_weights, w_table, w_weights

    def __buildOptimizer(self):
        a_input = tf.placeholder("float", [None, self.env.BOARD_SIZE])  # 预测的动作(落子位置)
        y_input = tf.placeholder("float", [None])  # 实际的"最佳"动作

        # 定义优化器
        b_readout_action = tf.reduce_sum(tf.multiply(self.b_Qtable, a_input), reduction_indices=1)
        b_loss = tf.reduce_mean(tf.square(y_input - b_readout_action))
        b_optimizer = tf.train.AdamOptimizer(1e-3).minimize(b_loss)

        w_readout_action = tf.reduce_sum(tf.multiply(self.b_Qtable, a_input), reduction_indices=1)
        w_loss = tf.reduce_mean(tf.square(y_input - w_readout_action))
        w_optimizer = tf.train.AdamOptimizer(1e-3).minimize(w_loss)
        return a_input, y_input, b_optimizer, b_loss, w_optimizer, w_loss


def main():
    env = ReversiEnv()
    agent = DQN(env)
    agent.run()


if __name__ == '__main__':
    main()
