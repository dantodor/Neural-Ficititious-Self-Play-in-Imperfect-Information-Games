import sys, logging
import tensorflow as tf
import leduc.game as game
import leduc.player as player
import agent.agent as agent
import numpy as np
from gym import spaces


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def testsuit():

    with tf.Session() as sess:
        espace = spaces.Box(low=0, high=1, shape=(3,))
        observation_state = espace.shape[0]
        a = agent.Agent(sess, observation_state, observation_state, 1)
        # state = np.array([[1, 2, 4]])
        state = np.zeros((1, 3))
        state[0][1] = 5
        sess.run(tf.global_variables_initializer())
        print(a.predict(state))

        # p1 = player.Player()
        # p2 = player.Player()
        # _game = game.Game(p1, p2)
        # _game.init_game()
        # _game.play_game()


if __name__ == '__main__':
    testsuit()
