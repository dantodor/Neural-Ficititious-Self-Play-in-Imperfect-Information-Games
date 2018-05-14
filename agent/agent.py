from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam, SGD
import ConfigParser
import utils.replay_buffer as ReplayBuffer
import utils.ReservoirBuffer as ReservoirBuffer
import numpy as np
from keras.callbacks import TensorBoard
import random
import keras.backend as K
import tensorflow as tf
import time
import math
from keras.layers.advanced_activations import LeakyReLU


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

        self.exploitability = 0
        self.iteration = 0

        # init parameters
        self.minibatch_size = int(self.config.get('Agent', 'MiniBatchSize'))
        self.n_hidden = int(self.config.get('Agent', 'HiddenLayer'))
        self.lr_br = float(self.config.get('Agent', 'LearningRateBR'))
        self.lr_ar = float(self.config.get('Agent', 'LearningRateAR'))
        self.epsilon = float(self.config.get('Agent', 'Epsilon'))
        self.epsilon_min = float(self.config.get('Agent', 'EpsilonMin'))
        self.gamma = float(self.config.get('Agent', 'Gamma'))
        self.omega = float(self.config.get('Agent', 'Omega'))
        self.sgd_br = SGD(lr=self.lr_br)
        self.sgd_ar = SGD(lr=self.lr_ar)
        self.target_model_update_rate = int(self.config.get('Agent', 'TargetModelUpdateRate'))

        self.iteration = 0
        self.temp = (1 + 0.02 * np.sqrt(self.iteration))**(-1)

        # mixed anticipatory parameter
        self.eta = float(self.config.get('Agent', 'Eta'))

        # target network update counter
        self.target_br_model_update_count = 0

        # reinforcement learning memory
        self._rl_memory = ReplayBuffer.ReplayBuffer(int(self.config.get('Utils', 'Buffersize')),
                                                    int(self.config.get('Utils', 'Seed')))

        # supervised learning memory
        self._sl_memory = ReservoirBuffer.ReservoirBuffer(int(self.config.get('Utils', 'Buffersize')),
                                                          int(self.config.get('Utils', 'Seed')))

        # build average strategy model
        self.avg_strategy_model = self._build_avg_response_model()

        # build dqn aka best response model
        self.best_response_model = self._build_best_response_model()
        self.target_br_model = self._build_best_response_model()
        self.target_br_model.set_weights(self.best_response_model.get_weights())

        # build supervised learning model
        # self.average_response_model = self._build_avg_response_model()
        self.actions = np.zeros(3)

        self.played = 0
        self.reward = 0
        self.test_reward = 0
        self.game_step = 0

        # tensorBoard
        self.tensorboard_br = TensorBoard(log_dir='./logs/'+self.name+'rl', histogram_freq=0,
                                          write_graph=False, write_images=True)

        self.tensorboard_sl = TensorBoard(log_dir='./logs/'+self.name+'sl', histogram_freq=0,
                                          write_graph=False, write_images=True)

    def _build_best_response_model(self):
        def huber_loss(a, b, in_keras=True):
            error = a - b
            quadratic_term = error * error / 2
            linear_term = abs(error) - 1 / 2
            use_linear_term = (abs(error) > 1.0)
            if in_keras:
                # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
                use_linear_term = K.cast(use_linear_term, 'float32')
            return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term

        input_ = Input(shape=self.s_dim, name='input')
        hidden = Dense(self.n_hidden, activation='relu')(input_)
        out = Dense(3, activation='relu')(hidden)

        model = Model(inputs=input_, outputs=out, name="br-model")
        model.compile(loss=huber_loss, optimizer=self.sgd_br, metrics=['accuracy', 'mse'])
        return model

    def _build_avg_response_model(self):
        input_ = Input(shape=self.s_dim, name='input')
        hidden = Dense(self.n_hidden, activation='relu')(input_)
        out = Dense(3, activation='softmax')(hidden)

        model = Model(inputs=input_, outputs=out, name="ar-model")
        model.compile(loss='categorical_crossentropy', optimizer=self.sgd_ar, metrics=['accuracy', 'crossentropy'])
        return model

    def remember_best_response(self, state, action):
        self._sl_memory.add(state, action)

    def remember_for_rl(self, state, action, reward, nextstate, terminal):
        self._rl_memory.add(state, action, reward, nextstate, terminal)

    def act_best_response(self, state):
        if random.random() > self.epsilon:
            return self.best_response_model.predict(state)
        else:
            return np.random.rand(1, 1, 3)

    def play(self, policy, index, s2=None):
        if s2 is None:
            s, a, r, s2, t = self.env.get_state(index)
            self.reward += r
            if np.average(a) != 0:
                self.remember_for_rl(s, a, r, s2, t)
                self.game_step += 1
            if t:
                return t
        else:  # because it's the initial state
            t = False
        if not t:
            if policy == 'a':
                a = self.avg_strategy_model.predict(np.reshape(s2, (1, 1, 30)))
                self.env.step(a, index)
                self.played += 1
            else:
                a_t = self.act_best_response(np.reshape(s2, (1, 1, 30)))
                # Evaluating the boltzmann distribution over Q Values
                a = self.boltzmann(a_t)
                self.env.step(a_t, index)
                self.remember_best_response(s2, a_t)
                self.played += 1
        if self.game_step % 128 == 0:
            self.update_strategy()
        self.actions[np.argmax(a)] += 1
        return t

    def boltzmann(self, actions):
        dist = np.zeros((1, 1, 3))
        bottom = 0
        for k in range(len(actions[0][0])):
            bottom += np.exp(actions[0][0][k] / self.temp)
        for k in range(len(actions[0][0])):
            top = np.exp(actions[0][0][k] / self.temp)
            dist[0][0][k] = top / bottom
        return dist

    def play_test(self, policy, index, s2=None):
        if s2 is None:
            s, a, r, s2, t = self.env.get_state(index)
            self.test_reward += r
            if t:
                return t
        else:  # because it's the initial state
            t = False
        if not t:
            if policy == 'a':
                a = self.avg_strategy_model.predict(np.reshape(s2, (1, 1, 30)))
                self.env.step(a, index)
            else:
                a = self.best_response_model.predict(np.reshape(s2, (1, 1, 30)))
                self.env.step(a, index)
        return t

    @property
    def play_test_get_reward(self):
        return self.test_reward

    def play_test_init(self):
        self.test_reward = 0

    def update_strategy(self):
        self.update_avg_response_network()
        self.update_best_response_network()

    def sampled_actions(self):
        print("{} played {} times: Folds: {}, Calls: {}, Raises: {} - Reward: {}".format(self.name,
                                                                                         self.played,
                                                                                         self.actions[0],
                                                                                         self.actions[1],
                                                                                         self.actions[2],
                                                                                         self.reward))
        self.actions = np.zeros(3)
        self.played = 0

    def average_payoff_br(self):
        return np.average(self.exploitability)

    def update_best_response_network(self):
        """
        Trains the dqn aka best response network trough
        replay experiences.
        """

        if self._rl_memory.size() > self.minibatch_size:
            self.iteration += 1
            s_batch, a_batch, r_batch, s2_batch, t_batch = self._rl_memory.sample_batch(self.minibatch_size)

            target = self.target_br_model.predict(s_batch)

            # if self.iteration > 50:
            #     print(target)

            reward = []

            for k in range(int(self.minibatch_size)):
                if t_batch[k] is True:
                    reward.append(r_batch[k])
                else:
                    Q_next = np.max(self.target_br_model.predict(np.reshape(s2_batch[k], (1, 1, 30))))
                    target_ = r_batch[k] + self.gamma * Q_next
                    reward.append(target_)

            # Evaluate exploitability
            expl = []
            for k in range(len(target)):
                expl.append(np.max(target[k]))
            self.exploitability = np.average(expl)

            for k in range(int(self.minibatch_size)):
                target[0][0][np.argmax(a_batch[k])] = reward[k]

            self.best_response_model.fit(s_batch, target, epochs=2, verbose=0, callbacks=[self.tensorboard_br])

            self.iteration += 1

            self.temp = (1 + 0.02 * np.sqrt(self.iteration)) ** (-1)

            self.update_br_target_network()

            K.set_value(self.sgd_br.lr, self.lr_br / (1 + 0.003 * math.sqrt(self.iteration)))

            self.epsilon = self.epsilon ** 1/self.iteration

    def update_avg_response_network(self):
        """
        Trains average response network with mapping action to state
        """
        if self._sl_memory.size() > self.minibatch_size:
            s_batch, a_batch = self._sl_memory.sample_batch(self.minibatch_size)
            self.avg_strategy_model.fit(s_batch, np.reshape(a_batch, (128, 1, 3)),
                                        epochs=2,
                                        verbose=0,
                                        callbacks=[self.tensorboard_sl])

    def update_br_target_network(self):
        if self.target_br_model_update_count % self.target_model_update_rate == 0:
            weights = self.best_response_model.get_weights()
            target_weights = self.target_br_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = weights[i]
            self.target_br_model.set_weights(target_weights)
        self.target_br_model_update_count += 1
