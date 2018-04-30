from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import SGD
import ConfigParser
import utils.replay_buffer as ReplayBuffer
import numpy as np
from keras.callbacks import TensorBoard
import random


class Agent:
    """
    Agent which contains the model & strategy
    """

    def __init__(self, sess, state_dim, action_dim):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.config = ConfigParser.ConfigParser()
        self.config.read("./config.ini")

        # init parameters
        self.minibatch_size = int(self.config.get('Agent', 'MiniBatchSize'))
        self.n_hidden = int(self.config.get('Agent', 'HiddenLayer'))
        self.lr_br = float(self.config.get('Agent', 'LearningRateBR'))
        self.lr_ar = float(self.config.get('Agent', 'LearningRateAR'))
        self.epsilon = float(self.config.get('Agent', 'Epsilon'))
        self.epsilon_min = float(self.config.get('Agent', 'EpsilonMin'))
        self.epsilon_decay = float(self.config.get('Agent', 'EpsilonDecay'))
        self.gamma = float(self.config.get('Agent', 'Gamma'))

        # mixed anticipatory parameter
        self.eta = float(self.config.get('Agent', 'Eta'))

        # reinforcement learning memory
        self._rl_memory = ReplayBuffer.ReplayBuffer(int(self.config.get('Utils', 'Buffersize')),
                                                    int(self.config.get('Utils', 'Seed')))

        # supervised learning memory
        self._sl_memory = ReplayBuffer.ReplayBuffer(int(self.config.get('Utils', 'Buffersize')),
                                                    int(self.config.get('Utils', 'Seed')))

        # build dqn aka best response model
        self.best_response_model = self._build_best_response_model()
        self.target_br_model = self._build_best_response_model()

        # build average strategy model
        self.avg_strategy_model = self._build_avg_response_model()

        # build supervised learning model
        # self.average_response_model = self._build_avg_response_model()

        # tensorborad
        self.tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                                       write_graph=True, write_images=True)

    def _build_best_response_model(self):
        """
        Initiates a DQN Agent which handles the reinforcement part of the fsp
        algorithm.

        :return:
        """
        x, y = self.s_dim
        input_shape = self.s_dim[1:]
        input_ = Input(shape=input_shape, name='input')
        hidden = Dense(self.n_hidden, activation='relu')(input_)
        out = Dense(3, activation='sigmoid')(hidden)

        model = Model(inputs=input_, outputs=out, name="br-model")
        model.compile(loss='mse', optimizer=SGD(lr=self.lr_br))
        return model

    def _build_avg_response_model(self):
        """
        Average response network, which learns via stochastic gradient descent on loss.

        :return:
        """
        input_shape = self.s_dim[1:]
        input_ = Input(shape=input_shape, name='input')
        hidden = Dense(self.n_hidden, activation='relu')(input_)
        out = Dense(3, activation='sigmoid')(hidden)

        model = Model(inputs=input_, outputs=out, name="ar-model")
        model.compile(loss='mean_squared_logarithmic_error', optimizer=SGD(lr=self.lr_ar))
        return model

    def remember_by_strategy(self, state, action, reward, nextstate, terminal, is_avg_stratey):
        if is_avg_stratey:
            self._sl_memory.add(state, action, reward, nextstate, terminal)
        else:
            self._rl_memory.add(state, action, reward, nextstate, terminal)

    def act(self, state):
        if random.random() > self.eta:
            # Append strategy information: True -> avg strategy is played
            return self.avg_strategy_model.predict(state), True
        else:
            return self.best_response_model.predict(state), False

    def update_strategy(self):
        self.update_best_response_network()
        self.update_avg_response_network()
        pass

    def update_best_response_network(self):
        """
        Trains the dqn aka best response network trough
        replay experiences.
        """

        if self._rl_memory.size() > self.minibatch_size:
            s_batch, a_batch, r_batch, s2_batch, t_batch = self._rl_memory.sample_batch(self.minibatch_size)

            for k in range(int(self.minibatch_size)):
                target = []
                if t_batch[k] == 1:
                    target = r_batch[k]
                else:
                    target = r_batch[k] + self.gamma * np.amax(self.best_response_model.predict(s2_batch[k]))

                target_f = self.best_response_model.predict(s_batch[k])

                target_f[0][a_batch[k]] = target

                self.best_response_model.fit(s_batch[k], target_f, epochs=1, verbose=2, callbacks=[self.tensorboard])
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def update_avg_response_network(self):
        """
        Trains average response network with mapping action to state
        """

        if self._sl_memory.size() > self.minibatch_size:
            s_batch, a_batch, _, _, _ = self._sl_memory.sample_batch(self.minibatch_size)
            for k in range(int(self.minibatch_size)):
                print("This is what i pass: {} {}".format(s_batch[k], a_batch[k]))
                self.avg_strategy_model.fit(s_batch[k], a_batch[k], verbose=2, callbacks=[self.tensorboard])

    def update_br_target_network(self):
        self.target_br_model.set_weights(self.best_response_model.get_weights())

