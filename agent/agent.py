from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam, SGD
import ConfigParser
import utils.replay_buffer as ReplayBuffer
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
        self.epsilon_decay = float(self.config.get('Agent', 'EpsilonDecay'))
        self.gamma = float(self.config.get('Agent', 'Gamma'))
        self.omega = float(self.config.get('Agent', 'Omega'))
        self.adam = SGD(lr=0.05)


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
        self.actions = np.zeros(3)

        self.played = 0
        self.reward = 0
        self.test_reward = 0

        self.game_step = 0

        self.avg_has_learned = False

        # tensorborad
        self.tensorboard_br = TensorBoard(log_dir='./logs/'+self.name+'rl', histogram_freq=0,
                                          write_graph=False, write_images=True)

        self.tensorboard_sl = TensorBoard(log_dir='./logs/'+self.name+'sl', histogram_freq=0,
                                          write_graph=False, write_images=True)

    def _build_best_response_model(self):
        input_ = Input(shape=self.s_dim, name='input')
        hidden = Dense(self.n_hidden, activation='relu')(input_)
        out = Dense(3, activation='linear')(hidden)

        model = Model(inputs=input_, outputs=out, name="br-model")
        model.compile(loss='mean_squared_error', optimizer=self.adam, metrics=['accuracy'])
        return model

    def _build_avg_response_model(self):
        input_ = Input(shape=self.s_dim, name='input')
        hidden = Dense(self.n_hidden, activation='relu')(input_)
        out = Dense(3, activation='softmax')(hidden)

        model = Model(inputs=input_, outputs=out, name="ar-model")
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=self.lr_ar), metrics=['accuracy', 'crossentropy'])
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
            br = self.best_response_model.predict(state)
            if br[0][0][0] == br[0][0][1] == br[0][0][2]:
                print("TRUE")
                return np.random.rand(1, 1, 3)
            else:
                return br
        else:
            return np.random.rand(1, 1, 3)

    def act_average_response(self, state):
        pred = self.avg_strategy_model.predict(state)
        return pred

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
        print("{} played {} times: Folds: {}, Calls: {}, Raises: {} - Reward: {}".format(self.name, self.played,
                                                                                         self.actions[0], self.actions[1],
                                                                                         self.actions[2], self.reward))
        self.actions = np.zeros(3)

        self.played = 0

    def average_payoff_br(self):
        # if self._rl_memory.size() > self.minibatch_size:
        #     s_batch, _, _, _, _ = self._rl_memory.sample_batch(self.minibatch_size)
        #     expl = []
        #     for k in range(int(len(s_batch))):
        #         expl.append(np.max(self.best_response_model.predict(np.reshape(s_batch[k], (1, 1, 30)))))
            # if np.average(expl) > 1:
            #     print self.best_response_model.predict(s_batch)
            #     time.sleep(60)
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
                    target_ = r_batch[k] * self.gamma * Q_next
                    reward.append(target_)

            # Evaluate exploitability
            expl = []
            for k in range(len(target)):
                expl.append(np.max(target[k]))
            self.exploitability = np.average(expl)


            for k in range(int(self.minibatch_size)):
                # print("{} for action {} got reward: {}".format(self.name, np.argmax(a_batch[k]), reward[k]))
                target[0][0][np.argmax(a_batch[k])] = reward[k]



            self.best_response_model.fit(s_batch, target, epochs=2, verbose=0, callbacks=[self.tensorboard_br])

            self.iteration += 1

            self.temp = (1 + 0.02 * np.sqrt(self.iteration)) ** (-1)

            self.update_br_target_network()

            K.set_value(self.adam.lr, 0.05 / (1 + 0.003 * math.sqrt(self.iteration)))

            self.epsilon = self.epsilon ** 1/self.iteration

    def update_avg_response_network(self):
        """
        Trains average response network with mapping action to state
        """

        if self._sl_memory.size() > self.minibatch_size:
            s_batch, a_batch, _, _, _ = self._sl_memory.sample_batch(self.minibatch_size)
            if self.avg_has_learned is False:
                print("{} adapts best response from now on.".format(self.name))
                self.avg_has_learned = True
            self.avg_strategy_model.fit(s_batch, np.reshape(a_batch, (128, 1, 3)), epochs=2, verbose=0, callbacks=[self.tensorboard_sl])

    def update_br_target_network(self):
        # Update target model network softly
        # main_model_weights = self.omega * np.asarray(self.best_response_model.get_weights())
        # target_model_weights = self.target_br_model.get_weights()
        # target_model_weights += main_model_weights
        if self.target_br_model_update_count % 300 == 0:
            self.target_br_model.set_weights(self.best_response_model.get_weights())
        self.target_br_model_update_count += 1






