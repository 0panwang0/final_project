import tensorflow as tf
import numpy as np
import random
from collections import deque
import gym
import time

GAMMA = 0.9  # targetQ保留率
INITIAL_EPSILON = 0.1  # 起始随机游走概率
FINAL_EPSILON = 0.01  # 最终随机游走概率，也就是说概率不能比这个还低
OBSERVE = 1000  # 先观察OBSERVE次，然后再训练
REPLAY_MEMORY = 10000  # 经验回放缓存大小
BATCH_SIZE = 200  # 每一批的训练量
TARGET_Q_STEP = 100  # 目标网络同步的训练次数
VALIDATE = 1000  # 每VALIDATE次查看一次训练效果

ENV_NAME = 'reversi-v0'  # 黑白棋环境名称
EPISODE = 10000  # 比赛次数
SAVE_EPISODE = 1000  # 每比赛SAVE_EPOCH次保存一次模型

# 棋子颜色
BLACK = -1
WHITE = 1
MY_COLOR = BLACK  # 我的棋子的颜色
OPP_COLOR = WHITE  # 对手棋子的颜色


class DQN:
    def __init__(self, env):
        self.env = env  # 黑白棋环境
        self.buffer = deque(maxlen=REPLAY_MEMORY)  # 经验池
        self.epsilon = INITIAL_EPSILON  # 随机游走率
        self.hide_layer_nums = 64  # 隐藏层数量
        self.sess = tf.InteractiveSession()  # 会话
        self.cnt = 1  # 计数器，储存训练次数

        # 输入层
        self.board_input = tf.placeholder("float", [None, self.env.BOARD_SIZE])

        # Q网络
        self.Qtable, self.Q_Weihgts = self.createNetwork()
        self.TargetQtable, self.TargetQ_Weights = self.createNetwork()

        # 定义优化器
        self.action_input, self.y_input, self.optimizer, self.loss = self.buildOptimizer()

    def createNetwork(self):
        # 输入层权重
        W1 = self.weight_variable([self.env.BOARD_SIZE, self.hide_layer_nums])
        b1 = self.bias_variable([self.hide_layer_nums])
        # 隐藏层权重
        W2 = self.weight_variable([self.hide_layer_nums, self.env.BOARD_SIZE])
        b2 = self.bias_variable([self.env.BOARD_SIZE])
        # 定义隐藏层
        h_layer = tf.nn.relu(tf.matmul(self.board_input, W1) + b1)
        # 定义table
        table = tf.matmul(h_layer, W2) + b2
        # 保存权重
        weights = [W1, b1, W2, b2]

        return table, weights

    def copyWeightsToTarget(self):
        """
        targetQ <- Q，详见2015年的那篇论文
        """
        for i in range(len(self.Q_Weihgts)):
            self.sess.run(tf.assign(self.TargetQ_Weights[i], self.Q_Weihgts[i]))

    def buildOptimizer(self):
        a_input = tf.placeholder("float", [None, self.env.BOARD_SIZE])  # 预测的动作(落子位置)
        y_input = tf.placeholder("float", [None])  # 实际的"最佳"动作

        # 定义优化器
        readout_action = tf.reduce_sum(tf.multiply(self.Qtable, a_input), reduction_indices=1)
        loss = tf.reduce_mean(tf.square(y_input - readout_action))
        optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
        return a_input, y_input, optimizer, loss

    def trainNetwork(self):
        minibatch = random.sample(self.buffer, BATCH_SIZE)
        # board, action, reward, next_board
        b_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        n_b_batch = [d[3] for d in minibatch]

        y_batch = []
        Qtable_batch = self.Qtable.eval(feed_dict={self.board_input: n_b_batch})
        for i in range(BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal == self.env.GAMING:
                y_batch.append(r_batch[i] + GAMMA * np.max(Qtable_batch[i]))
            else:
                y_batch.append(r_batch[i])
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={
            self.y_input: y_batch,
            self.action_input: a_batch,
            self.board_input: b_batch
        })
        return loss

    def epsilon_greedy(self, board, my_color):
        """
        这个是用于训练的
        """

        Qtable = self.Qtable.eval(feed_dict={
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

        # 随机游走率随着迭代次数逐渐降低
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000

        if random.random() <= self.epsilon:  # 随机游走
            return random.choice(valid_action)
        else:  # 取权重最大的那个落子位置
            return np.argmax(Qtable)

    def action(self, board, my_color):
        """
        这个是用于检验效果的，所以不用随机游走
        """

        Qtable = self.Qtable.eval(
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

    def run(self):
        # ----------------------------initial----------------------------
        saver = tf.train.Saver()  # 储存器
        checkpoint = tf.train.get_checkpoint_state('save/')

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("加载之前的模型")
        else:
            print("未发现之前的模型，重新开始训练")

        self.sess.run(tf.global_variables_initializer())
        if not (checkpoint and checkpoint.model_checkpoint_path):
            self.copyWeightsToTarget()
        # ----------------------------initial----------------------------

        for episode in range(EPISODE):
            print('--------------EPISODE:', episode, '--------------')

            board = self.env.reset()  # 获得棋盘
            color = MY_COLOR  # 执棋方的颜色，初始为我的颜色
            # 开始训练
            turn = 0  # 控制执棋方
            while True:
                # 自己下一步棋
                action = self.epsilon_greedy(board, color)  # 获取下一步的行动
                new_board, reward, terminal = self.env.step((int(action), color, MY_COLOR))  # 根据行动得到下一状态、奖励、游戏是否终止三个参数

                if action != -1:
                    action_list = np.zeros(self.env.BOARD_SIZE)
                    action_list[action] = 1
                    # 存放到经验池
                    self.buffer.append([board, action_list, reward, new_board, terminal])
                    # 当经验池中的数据足够多时开始训练
                    if len(self.buffer) > OBSERVE:
                        self.cnt += 1
                        loss = self.trainNetwork()
                        print("loss:", loss)
                        # 同步目标网络
                        if self.cnt % TARGET_Q_STEP == 0:
                            self.copyWeightsToTarget()

                # 双方轮流下棋
                color = OPP_COLOR if turn % 2 == 0 else MY_COLOR
                board = new_board  # 更新棋盘

                if terminal != self.env.GAMING:  # 结束比赛
                    winner = "黑方" if terminal == self.env.BLACK else "白方" if terminal == self.env.WHITE else "平局"
                    print('胜利方为' + winner)
                    break
                turn += 1

            # 保存模型
            if episode % SAVE_EPISODE == 0:
                saver.save(self.sess, 'save/', global_step=episode)

            # 验证
            if (episode + 1) % VALIDATE == 0:
                board = self.env.reset()
                color = MY_COLOR  # 执棋方的颜色，初始为我的颜色
                turn = 0  # 控制执棋方

                while True:
                    print("------------------------step {}--------------------------".format(turn + 1))
                    self.env.render()
                    action = self.action(board, color)
                    new_board, reward, terminal = self.env.step((int(action), color, MY_COLOR))

                    # 双方轮流下棋
                    color = OPP_COLOR if turn % 2 == 0 else MY_COLOR
                    board = new_board  # 更新棋盘

                    time.sleep(1)
                    turn += 1
                    if terminal != self.env.GAMING:  # 结束比赛
                        print("------------------------step {}--------------------------".format(turn + 1))
                        self.env.render()
                        winner = "黑方" if terminal == self.env.BLACK else "白方" if terminal == self.env.WHITE else "平局"
                        print('胜利方为' + winner)
                        break


def main():
    env = gym.make(ENV_NAME)
    agent = DQN(env)
    agent.run()


if __name__ == '__main__':
    main()
