from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import ConfigParser
import utils.replay_buffer as ReplayBuffer
import numpy as np
import random


class Agent:
    """
    Agent which contains the model & strategy
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate

        self.config = ConfigParser.ConfigParser()
        self.config.read("./config.ini")
        self.br_hidden_1 = int(self.config.get('Agent', 'BRHidden1'))
        self.br_hidden_2 = int(self.config.get('Agent', 'BRHidden2'))
        self.ar_hidden_1 = int(self.config.get('Agent', 'ARHidden1'))
        self.ar_hidden_2 = int(self.config.get('Agent', 'ARHidden2'))

        self.minibatch_size = int(self.config.get('Agent', 'MiniBatchSize'))

        # init parameters
        self.epsilon = float(self.config.get('Agent', 'Epsilon'))
        self.gamma = float(self.config.get('Agent', 'Gamma'))

        # reinforcement learning memory
        self._rl_memory = ReplayBuffer.ReplayBuffer(int(self.config.get('Utils', 'Buffersize')), int(self.config.get('Utils', 'Seed')))

        # supervised learning memory
        self._sl_memory = ReplayBuffer.ReplayBuffer(int(self.config.get('Utils', 'Buffersize')), int(self.config.get('Utils', 'Seed')))

        # build dqn aka best response model
        self.best_response_model = self._build_best_response_model()

        # build supervised learning model
        self.average_response_model = self._build_avg_response_model()

    def _build_best_response_model(self):
        """
        Initiates a DQN Agent which handles the reinforcement part of the fsp
        algorithm.

        :return:
        """
        model = Sequential()
        model.add(Dense(self.br_hidden_1, input_dim=self.s_dim, activation='relu'))
        model.add(Dense(self.br_hidden_2, activation='relu'))
        model.add(Dense(self.a_dim, activation='sigmoid'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_avg_response_model(self):
        """
        TODO: Find a supervised learning algorithm -> Backpropagation?

        :return:
        """
        model = Sequential()
        model.add(Dense(self.br_hidden_1, input_dim=self.s_dim, activation='relu'))
        model.add(Dense(self.br_hidden_2, activation='relu'))
        model.add(Dense(self.a_dim, activation='sigmoid'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember_opponent_behaviour(self, state, action, reward, nextstate, terminal):
        self._rl_memory.add(state, action, reward, nextstate, terminal)

    def br_network_act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.a_dim)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def update_best_response_network(self):
        """
        Trains the dqn aka best response network trough
        replay experiences.

        :return:
        """

        s_batch, a_batch, r_batch, s2_batch, t_batch = self._rl_memory.sample_batch(self.minibatch_size)

        for k in range(int(self.minibatch_size)):





    def update_avg_response_network(self):
        pass

    def predict(self, inputs):
        return self.sess.run(self.a_out, feed_dict={
            self.a_inputs: inputs
        })
