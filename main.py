import sys, logging
import tensorflow as tf
import leduc.env as leduc
import agent.agent as agent
import numpy as np
from gym import spaces
import argparse
import ConfigParser
import random

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# Configuration
Config = ConfigParser.ConfigParser()
Config.read("./config.ini")


def train(sess, env, args, player1, player2):

    # initialize tensorflow variables
    sess.run(tf.global_variables_initializer())

    for i in range(int(Config.get('Common', 'MaxEpisodes'))):

        # choose a dealer randomly
        dealer = random.randint(0, 1)

        # reset env
        env.reset()

        # get initial state
        p1_s = env.init_state(0)
        p2_s = env.init_state(1)

        if dealer == 0:  # player1 is dealer
            # TODO: Fix return of env.step to -> p1_sOld, p1_a, r, p1_s, t, i
            p1_a = player1.predict(p1_s)
            p1_s, r, t, i = env.step(p1_a)

            p2_a = player2.predict(p2_s)
            p2_s, r, t, i = env.step(p2_a)





def main(args):

    with tf.Session() as sess:

        env = leduc.Env()
        np.random.seed(int(Config.get('Utils', 'Seed')))
        tf.set_random_seed(int(Config.get('Utils', 'Seed')))

        # initialize dimensions:
        state_dim = env.observation_space
        action_dim = env.action_space

        # initialize players
        player1 = agent.Agent(sess, state_dim, action_dim, float(Config.get('Agent', 'LearningRate')))
        player2 = agent.Agent(sess, state_dim, action_dim, float(Config.get('Agent', 'LearningRate')))

        train(sess, env, args, player1, player2)



def testfunc(args):

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

    print("NFSP by David Joos")
    parser = argparse.ArgumentParser(description='Provide arguments for NFSP agent.')

    parser.add_argument('--testfunc', help='starts testfunc function instead of main', action='store_true')

    args = vars(parser.parse_args())

    if args['testfunc'] is True:
        testfunc(args)
    else:
        main(args)


