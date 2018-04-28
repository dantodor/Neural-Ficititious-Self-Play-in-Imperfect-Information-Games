from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import ConfigParser
import utils.replay_buffer as ReplayBuffer
import numpy as np
from keras.callbacks import TensorBoard
import random

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)

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
        self.epsilon_min = float(self.config.get('Agent', 'EpsilonMin'))
        self.epsilon_decay = float(self.config.get('Agent', 'EpsilonDecay'))
        self.gamma = float(self.config.get('Agent', 'Gamma'))

        # reinforcement learning memory
        self._rl_memory = ReplayBuffer.ReplayBuffer(int(self.config.get('Utils', 'Buffersize')), int(self.config.get('Utils', 'Seed')))

        # supervised learning memory
        self._sl_memory = ReplayBuffer.ReplayBuffer(int(self.config.get('Utils', 'Buffersize')), int(self.config.get('Utils', 'Seed')))

        # build dqn aka best response model
        self.best_response_model = self._build_best_response_model()

        # build supervised learning model
        # self.average_response_model = self._build_avg_response_model()

    def _build_best_response_model(self):
        """
        Initiates a DQN Agent which handles the reinforcement part of the fsp
        algorithm.

        :return:
        """
        # model = Sequential()
        # model.add(Dense(self.br_hidden_1, input_dim=self.s_dim, activation='relu'))
        # model.add(Dense(self.br_hidden_2, activation='relu'))
        # model.add(Dense(self.a_dim, activation='sigmoid'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        x, y = self.s_dim
        input_shape = self.s_dim[1:]
        input_ = Input(shape=input_shape, name='input')
        hidden1 = Dense(self.br_hidden_1, activation='relu')(input_)
        hidden2 = Dense(self.br_hidden_2, activation='relu')(hidden1)
        out = Dense(3, activation='sigmoid')(hidden2)

        model = Model(inputs=input_, outputs=out, name="br-model")
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
        #if np.random.rand() <= self.epsilon:
        #    return random.randrange(self.a_dim)
        act_values = self.best_response_model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def update_best_response_network(self):
        """
        Trains the dqn aka best response network trough
        replay experiences.

        :return:
        """

        if self._rl_memory.size() > self.minibatch_size:
            s_batch, a_batch, r_batch, s2_batch, t_batch = self._rl_memory.sample_batch(self.minibatch_size)

            for k in range(int(self.minibatch_size)):
                target = []
                if t_batch[k] == 1:
                    target = r_batch[k]
                else:
                    target = r_batch[k] + self.gamma * np.amax(self.best_response_model.predict(s2_batch[k]))

                target_f = self.best_response_model.predict(s_batch[k], batch_size=1)

                target_f[0][a_batch[k]] = target

                self.best_response_model.fit(s_batch[k], target_f, epochs=1, verbose=0, callbacks=[tensorboard])
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def update_avg_response_network(self):
        pass

    def evaluate_best_response_network(self):
        if self._rl_memory.size() > self.minibatch_size:
            s_batch, a_batch, r_batch, s2_batch, t_batch = self._rl_memory.sample_batch(self.minibatch_size)
            s_batch2, a_batch2, r_batch2, s2_batch2, t_batch2 = self._rl_memory.sample_batch(self.minibatch_size)
            for k in range(self.minibatch_size):
                eval_ = self.best_response_model.evaluate(s_batch[k], s2_batch[k])
            print (eval_)

    def predict(self, inputs):
        return self.sess.run(self.a_out, feed_dict={
            self.a_inputs: inputs
        })
