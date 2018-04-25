import sys, logging
import tensorflow as tf
import leduc.env as leduc
import leduc.player as player
import agent.agent as agent
import numpy as np
from gym import spaces


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def testsuit():

    env = leduc.Env()

    with tf.Session() as sess:


        espace = spaces.Box(low=0, high=1, shape=(3,))
        observation_state = espace.shape[0]
        a = agent.Agent(sess, observation_state, observation_state, 1)
        # state = np.array([[1, 2, 4]])
        # state = np.zeros((1, 3))
        # state[0][1] = 5
        sess.run(tf.global_variables_initializer())

        env.reset()

        #BOT1
        state, reward, term, info = env.init_state(0)
        state = np.array([state])
        action1 = a.predict(state)
        action1 = [0, 0, 1]
        print("Action BOT1: {}".format(action1))

        #BOT2
        state, reward, term, info = env.init_state(1)
        state = np.array([state])
        action2 = a.predict(state)
        action2 = [0, 0, 1]
        print("Action BOT2: {}".format(action2))

        terminal = 0
        while terminal <= 1:
            print("="*30)
            print("BOT1:")
            state, reward, term, info = env.step(action1, 0)
            print("state: {}".format(state))
            print("reward: {}".format(reward))
            print("terminal: {}".format(term))
            print("info: {}".format(info))
            print("=" * 30)
            print("BOT2:")
            state, reward, term2, info = env.step(action2, 1)
            print("state: {}".format(state))
            print("reward: {}".format(reward))
            print("terminal: {}".format(term2))
            print("info: {}".format(info))

            terminal += term
            terminal += term2




if __name__ == '__main__':
    testsuit()
