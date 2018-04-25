from keras.layers import Input, Dense
import ConfigParser


class Agent:
    """
    Agent which contains the model & strategy
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate

        config = ConfigParser.ConfigParser()
        config.read("./config.ini")
        self.br_hidden_1 = int(config.get('Agent', 'BRHidden1'))
        self.br_hidden_2 = int(config.get('Agent', 'BRHidden2'))
        self.ar_hidden_1 = int(config.get('Agent', 'ARHidden1'))
        self.ar_hidden_2 = int(config.get('Agent', 'ARHidden2'))

        # best response network:
        self.b_inputs, self.b_out = self.init_best_response_network()

        # avg response network:
        self.a_inputs, self.a_out = self.init_avg_response_network()

    def init_best_response_network(self):
        inputs = Input(shape=[self.a_dim, ])
        hidden1 = Dense(self.br_hidden_1, activation='relu')(inputs)
        hidden2 = Dense(self.br_hidden_2, activation='relu')(hidden1)
        out = Dense(self.a_dim, activation='softmax')(hidden2)
        return inputs, out

    def init_avg_response_network(self):
        inputs = Input(shape=[self.a_dim, ])
        hidden1 = Dense(self.ar_hidden_1, activation='relu')(inputs)
        hidden2 = Dense(self.ar_hidden_2, activation='relu')(hidden1)
        out = Dense(self.a_dim, activation='softmax')(hidden2)
        return inputs, out

    def update_best_response_network(self):
        pass

    def update_avg_response_network(self):
        pass

    def predict(self, inputs):
        return self.sess.run(self.a_out, feed_dict={
            self.a_inputs: inputs
        })
