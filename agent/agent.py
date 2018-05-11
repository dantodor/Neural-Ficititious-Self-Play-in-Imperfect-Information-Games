from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import ConfigParser
import utils.replay_buffer as ReplayBuffer
import numpy as np
from keras.callbacks import TensorBoard
import random
import keras.backend as K
import time


class Agent:
    """
    Agent which contains the model & strategy
    """

    def __init__(self, sess, state_dim, action_dim, name, env):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.name = name
        self.env = env

        self.config = ConfigParser.ConfigParser()
        self.config.read("./config.ini")

        self.exploitability = []
        self.iteration = 0

        # init parameters
        self.minibatch_size = int(self.config.get('Agent', 'MiniBatchSize'))
        self.n_hidden = int(self.config.get('Agent', 'HiddenLayer'))
        self.lr_br = float(self.config.get('Agent', 'LearningRateBR'))
        self.lr_ar = float(self.config.get('Agent', 'LearningRateAR'))
        self.epsilon = float(self.config.get('Agent', 'Epsilon'))
        self.epsilon_min = float(self.config.get('Agent', 'EpsilonMin'))
        self.epsilon_decay = float(self.config.get('Agent', 'EpsilonDecay'))
        self.gamma = float(self.config.get('Agent', 'Gamma'))
        self.omega = float(self.config.get('Agent', 'Omega'))

        self.iteration = 0

        # mixed anticipatory parameter
        self.eta = float(self.config.get('Agent', 'Eta'))

        # target network update counter
        self.target_br_model_update_count = 0

        # reinforcement learning memory
        self._rl_memory = ReplayBuffer.ReplayBuffer(int(self.config.get('Utils', 'Buffersize')),
                                                    int(self.config.get('Utils', 'Seed')))

        # supervised learning memory
        self._sl_memory = ReplayBuffer.ReplayBuffer(int(self.config.get('Utils', 'Buffersize')),
                                                    int(self.config.get('Utils', 'Seed')))

        # build average strategy model
        self.avg_strategy_model = self._build_avg_response_model()

        # build dqn aka best response model
        self.best_response_model = self._build_best_response_model()
        self.target_br_model = self._build_best_response_model()
        self.target_br_model.set_weights(self.best_response_model.get_weights())

        # build supervised learning model
        # self.average_response_model = self._build_avg_response_model()

        # tensorborad
        self.tensorboard_br = TensorBoard(log_dir='./logs/'+self.name+'rl', histogram_freq=0,
                                       write_graph=False, write_images=True)

        self.tensorboard_sl = TensorBoard(log_dir='./logs/'+self.name+'sl', histogram_freq=0,
                                       write_graph=False, write_images=True)


    def _build_best_response_model(self):
        """
        Initiates a DQN Agent which handles the reinforcement part of the fsp
        algorithm.

        :return:
        """

        def expl(y_true, y_pred):
            return K.mean(y_pred)

        input_ = Input(shape=self.s_dim, name='input')
        hidden = Dense(self.n_hidden, activation='relu')(input_)
        out = Dense(3, activation='softmax')(hidden)

        model = Model(inputs=input_, outputs=out, name="br-model")
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.lr_br), metrics=['accuracy', expl])
        # model = Sequential()
        # model.add(Dense(self.n_hidden, activation='relu', input_dim=int(self.s_dim[1:][0])))
        # model.add(Dense(3, activation='softmax'))
        # model.compile(optimizer=SGD(lr=self.lr_br),
        #               loss='mse',
        #               metrics=['accuracy'])
        return model

    def _build_avg_response_model(self):
        """
        Average response network, which learns via stochastic gradient descent on loss.

        :return:
        """
        input_ = Input(shape=self.s_dim, name='input')
        hidden = Dense(self.n_hidden, activation='relu')(input_)
        out = Dense(3, activation='softmax')(hidden)

        model = Model(inputs=input_, outputs=out, name="ar-model")
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr_ar), metrics=['accuracy'])
        # model = Sequential()
        # model.add(Dense(self.n_hidden, activation='relu', input_dim=int(self.s_dim[1:][0])))
        # model.add(Dense(3, activation='softmax'))
        # model.compile(optimizer=SGD(lr=self.lr_ar),
        #               loss='categorical_crossentropy',
        #               metrics=['accuracy'])
        return model

    def remember_by_strategy(self, state, action, reward, nextstate, terminal, is_avg_stratey):
        if is_avg_stratey:
            self._rl_memory.add(state, action, reward, nextstate, terminal)
        else:
            self._rl_memory.add(state, action, reward, nextstate, terminal)
            self._sl_memory.add(state, action, reward, nextstate, terminal)

    def remember_best_response(self, state, action):
        self._sl_memory.add(state, action, None, None, None)

    def remember_for_rl(self, state, action, reward, nextstate, terminal):
        self._rl_memory.add(state, action, reward, nextstate, terminal)

    def act(self, state):
        if random.random() > self.eta:
            # Append strategy information: True -> avg strategy is played
            return self.avg_strategy_model.predict(state), True
        else:
            if random.random() > self.epsilon:
                return self.best_response_model.predict(state), False
            else:
                return np.random.rand(1, 3), False
        # return np.random.rand(1, 3), False

    def act_best_response(self, state):
        if random.random() > self.epsilon:
            return self.best_response_model.predict(state)
        else:
            return np.random.rand(1, 3)

    def act_average_response(self, state):
        return self.avg_strategy_model.predict(state)

    def play(self, policy, index, s=None):
        if s is None:
            s, a, r, s2, t = self.env.get_state(index)
            self.remember_for_rl(s, a, r, s2, t)
            if t:
                return t
        else:  # because it's the initial state
            t = False
        if not t:
            if policy == 'avg':
                a = self.avg_strategy_model.predict(np.reshape(s, (1, 1, 30)))
                self.env.step(a, index)
            else:
                a = self.best_response_model.predict(np.reshape(s, (1, 1, 30)))
                self.env.step(a, index)
                self.remember_best_response(s, a)
        self.update_strategy()
        return t

    def update_strategy(self):
        self.update_avg_response_network()
        self.update_best_response_network()

    def average_payoff_br(self):
        if self._rl_memory.size() > self.minibatch_size:
            s_batch, _, _, _, _ = self._rl_memory.sample_batch(self.minibatch_size)
            expl = []
            for k in range(int(len(s_batch))):
                expl.append(np.max(self.best_response_model.predict(s_batch[k])))
            return np.average(expl)

    def update_best_response_network(self):
        """
        Trains the dqn aka best response network trough
        replay experiences.
        """

        if self._rl_memory.size() > self.minibatch_size:
            self.iteration += 1
            target = []
            s_batch, a_batch, r_batch, s2_batch, t_batch = self._rl_memory.sample_batch(self.minibatch_size)
            for k in range(int(self.minibatch_size)):
                if t_batch[k] is True:
                    target.append(r_batch[k])
                else:
                    target.append(r_batch[k] + self.gamma * np.amax(self.target_br_model.predict(np.reshape(s2_batch[k], (1, 1, 30)))))

            target_f = self.best_response_model.predict(s_batch)

            for k in range(self.minibatch_size):
                target_f[k][0][np.argmax(a_batch[k])] = target[k]

            self.best_response_model.fit(s_batch, target_f, epochs=2, verbose=0, callbacks=[self.tensorboard_br])
            self.iteration += 1
            self.update_br_target_network()

            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon ** 1/self.iteration

    def update_avg_response_network(self):
        """
        Trains average response network with mapping action to state
        """

        if self._sl_memory.size() > self.minibatch_size:
            s_batch, a_batch, _, _, _ = self._sl_memory.reservoir_sample(self.minibatch_size)
            self.avg_strategy_model.fit(s_batch, a_batch, epochs=2, verbose=0, callbacks=[self.tensorboard_sl])

    def update_br_target_network(self):
        # Update target model network softly
        # main_model_weights = self.omega * np.asarray(self.best_response_model.get_weights())
        # target_model_weights = self.target_br_model.get_weights()
        # target_model_weights += main_model_weights
        if self.target_br_model_update_count % 300 == 0:
            self.target_br_model.set_weights(self.best_response_model.get_weights())
        self.target_br_model_update_count += 1
