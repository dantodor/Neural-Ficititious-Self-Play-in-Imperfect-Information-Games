import sys, logging
import tensorflow as tf
import leduc.env as leduc
import agent.agent as agent
import numpy as np
from gym import spaces
import argparse
import ConfigParser
import random
import logging

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
log = logging.getLogger('')


# Configuration
Config = ConfigParser.ConfigParser()
Config.read("./config.ini")


def fsp(sess, env, args, player1, player2):

    # initialize tensorflow variables
    sess.run(tf.global_variables_initializer())

    for i in range(int(Config.get('Common', 'MaxEpisodes'))):

        # init temp vars
        dealer = 0
        wins_p1 = 0
        wins_p2 = 0

        for j in range(int(Config.get('Common', 'Episodes'))):

            env.reset()
            # Pass init state to players
            p1_s = env.init_state(0)
            p2_s = env.init_state(1)

            # Setting avg to false, will be overwritten later on
            p1_is_avg_strat = False
            p2_is_avg_strat = False

            # Randomly set dealer
            dealer = random.randint(0, 1)

            terminated = False
            first_round = True
            while not terminated:
                if dealer == 0:  # player1 is dealer
                    if first_round:
                        # Player1 follows predicted action on init state
                        p1_a, p1_is_avg_strat = player1.act(p1_s)
                        env.step(p1_a, 0)

                        # Player2
                        p2_s2, p2_a, p2_r, p2_t, p2_i = env.get_new_state(1)
                        p2_a, p2_is_avg_strat = player2.act(p2_s)
                        env.step(p2_a, 1)

                        first_round = False
                    else:
                        p1_s2, p1_a, p1_r, p1_t, p1_i = env.get_new_state(0)
                        player1.remember_by_strategy(p1_s, p1_a, p1_r, p1_s2, p1_t, p1_is_avg_strat)
                        p1_s = p1_s2
                        p1_a, p1_is_avg_strat = player1.act(p1_s)
                        env.step(p1_a)

                        p2_s2, p2_a, p2_r, p2_t, p2_i = env.get_new_state(1)
                        player2.remember_by_strategy(p2_s, p2_a, p2_r, p2_s2, p2_t, p2_is_avg_strat)
                        p2_s = p2_s2
                        p2_a, p2_is_avg_strat = player2.act(p2_s)
                        env.step(p2_a)

                    if p1_t == 1 and p2_t == 1:
                        terminated = True

                if dealer == 1:  # player2 is dealer
                    if first_round:
                        # Player2 follows predicted action on init state
                        p2_a, p2_is_avg_strat = player2.act(p2_s)
                        env.step(p2_a, 1)

                        # Player1
                        p1_s2, p1_a, p1_r, p1_t, p1_i = env.get_new_state(0)
                        p1_a, p1_is_avg_strat = player1.act(p2_s)
                        env.step(p1_a, 0)

                        first_round = False
                    else:
                        p2_s2, p2_a, p2_r, p2_t, p2_i = env.get_new_state(1)
                        player2.remember_by_strategy(p2_s, p2_a, p2_r, p2_s2, p2_t, p2_is_avg_strat)
                        p2_s = p2_s2
                        p2_a, p2_is_avg_strat = player2.act(p2_s)
                        env.step(p2_a, 1)

                        p1_s2, p1_a, p1_r, p1_t, p1_i = env.get_new_state(0)
                        player1.remember_by_strategy(p1_s, p1_a, p1_r, p1_s2, p1_t, p1_is_avg_strat)
                        p1_s = p1_s2
                        p1_a, p1_is_avg_strat = player1.act(p2_s)
                        env.step(p1_a, 0)

                    if p1_t == 1 and p2_t == 1:
                        terminated = True

        player1.update_strategy()
        player2.update_strategy()


def main(args):

    with tf.Session() as sess:

        env = leduc.Env()
        np.random.seed(int(Config.get('Utils', 'Seed')))
        tf.set_random_seed(int(Config.get('Utils', 'Seed')))

        # initialize dimensions:
        state_dim = env.dim_shape
        action_dim = env.dim_shape

        # initialize players
        player1 = agent.Agent(sess, state_dim, action_dim)
        player2 = agent.Agent(sess, state_dim, action_dim)

        # Start fictitious self play algorithm
        fsp(sess, env, args, player1, player2)


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


