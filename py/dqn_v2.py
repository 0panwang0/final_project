import tensorflow as tf
import numpy as np
import random
from collections import deque
import time
from reversi import ReversiEnv

GAMMA = 0.98
ENV_NAME = 'reversi-v0'
EPISODE = 1000
REPLAY_MEMORY = 1000
BATCH_SIZE = 256
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.005
BOARD_SIZE = 64
HIDDEN_SIZE = 44
DATATYPE = tf.float32

BLACK = -1
WHITE = 1


class DQN():

    def __init__(self, env):
        self.env = env
        self.color = BLACK

        self.replay_buffer = deque()
        self.buffer_length = 0

        self.time_step = 0
        self.epsilon = INITIAL_EPSILON

        self.create_network()
        self.create_training_method()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())


    def create_network(self):
        self.state_input = tf.placeholder(DATATYPE, [None, BOARD_SIZE])
        self.y_input = tf.placeholder(DATATYPE, [None])
        self.action_input = tf.placeholder(DATATYPE, [None, BOARD_SIZE])

        W1 = self.weight_variable([BOARD_SIZE, HIDDEN_SIZE])
        b1 = self.bias_variable([HIDDEN_SIZE])

        W2 = self.weight_variable([HIDDEN_SIZE, BOARD_SIZE])
        b2 = self.bias_variable([BOARD_SIZE])

        hidden = tf.nn.tanh(tf.matmul(self.state_input, W1) + b1)
        self.Q_value = tf.nn.tanh(tf.matmul(hidden, W2) + b2)


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape=shape, dtype=DATATYPE)
        return tf.Variable(initial, dtype=DATATYPE)


    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape, dtype=DATATYPE)
        return tf.Variable(initial, dtype=DATATYPE)


    def create_training_method(self):
        '''定义损失函数'''
        action_value = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), axis=1)
        self.loss = tf.reduce_mean(tf.square(self.y_input - action_value))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)


    def perceive(self, state, action, reward, next_state, done):
        one_hot = np.zeros(BOARD_SIZE)
        one_hot[action] = 1

        self.replay_buffer.append((state, one_hot, reward, next_state, done))
        self.buffer_length += 1

        loss = None
        if self.buffer_length > REPLAY_MEMORY:
            self.replay_buffer.popleft()
            self.buffer_length -= 1
        
        if self.buffer_length > BATCH_SIZE:
            loss = self.train_network()

        return loss


    def train_network(self):
        self.time_step += 1

        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        state = [b[0] for b in batch]
        one_hot = [b[1] for b in batch]
        reward = [b[2] for b in batch]
        next_state = [b[3] for b in batch]
        done = [b[4] for b in batch]

        y = []
        value = self.Q_value.eval(feed_dict={
            self.state_input : next_state,
        })
        for i in range(BATCH_SIZE):
            if done[i] == True:
                y.append(reward[i])
            else:
                # self.filter_value(value[i], board=next_state[i])
                y.append(reward[i] + GAMMA * np.max(value[i]))

        self.session.run(self.optimizer, feed_dict={
            self.state_input : state,
            self.action_input : one_hot,
            self.y_input : y,
        })

        loss = self.session.run(self.loss, feed_dict={
            self.state_input : state,
            self.action_input : one_hot,
            self.y_input : y,
        })

        return loss


    def epsilon_greedy(self, state):
        value = self.Q_value.eval(feed_dict={
            self.state_input : [state],
        })[0]

        valid_action = self.env.get_valid_pos(self.color)
        if not valid_action:
            return -1

        self.filter_value(value, valid_action)
        if random.random() <= self.epsilon:
            ret = random.choice(valid_action)
        else:
            ret = np.argmax(value)

        self.epsilon = (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        self.epsilon = max(self.epsilon, FINAL_EPSILON)

        return ret


    def action(self, state):
        value = self.Q_value.eval(feed_dict={
            self.state_input : [state],
        })[0]
        self.filter_value(value)

        return np.argmax(value)

    
    def filter_value(self, value, valid_action=None):
        '''把不能落子的位置的Q-value设置为-inf'''
        if valid_action == None:
            valid_action = self.env.get_valid_pos(self.color)
        valid_action = set(valid_action)
        for i in range(BOARD_SIZE):
            if i not in valid_action:
                value[i] = -100000


    def change_player(self):
        self.color = WHITE if self.color == BLACK else BLACK


    def reset_color(self):
        self.color = BLACK
    

def main():
    random.seed(int(time.time()))

    env = ReversiEnv()
    agent = DQN(env)

    for episode in range(EPISODE):
        print('---------- Episode #{} ----------'.format(episode))
        state1 = env.reset()
        agent.reset_color()
        first = True
        loss = None

        while True:
            '''黑方下棋'''
            action1 = agent.epsilon_greedy(state1)
            next_state2, reward1, winner = env.step((int(action1), agent.color, agent.color))
            done = True if winner != env.GAMING else False
            '''第一次下棋，state2不存在'''
            if not first:
                loss = agent.perceive(state2, action2, reward2 - reward1, next_state2, done)
            else:
                first = False
            state2 = next_state2
            agent.change_player()

            if done:
                agent.perceive(state1, action1, reward1 - reward2, next_state2, done)
                break

            if agent.time_step % 10 == 0 and loss:
                print('Loss : {}'.format(loss))

            '''白方下棋'''
            action2 = agent.epsilon_greedy(state2)
            next_state1, reward2, winner = env.step((int(action2), agent.color, agent.color))
            done = True if winner != env.GAMING else False
            loss = agent.perceive(state1, action1, reward1 - reward2, next_state1, done)
            state1 = next_state1
            agent.change_player()

            if done:
                agent.perceive(state2, action2, reward2 - reward1, next_state1, done)
                break

            if agent.time_step % 10 == 0 and loss:
                print('Loss : {}'.format(loss))


if __name__ == '__main__':
    main()