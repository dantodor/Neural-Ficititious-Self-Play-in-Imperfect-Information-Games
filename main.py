import sys, logging
import tensorflow as tf
import leduc.newenv as leduc
import agent.agent as agent
import numpy as np
from gym import spaces
import argparse
import ConfigParser
import random
import logging
import time
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
log = logging.getLogger('')


# Configuration
Config = ConfigParser.ConfigParser()
Config.read("./config.ini")


def fsp(sess, env, args, player1, player2):

    # initialize tensorflow variables
    sess.run(tf.global_variables_initializer())

    players = [player1, player2]

    for i in range(int(Config.get('Common', 'MaxEpisodes'))):
        for j in range(int(Config.get('Common', 'Episodes'))):

            env.reset()

            # Randomly set dealer
            dealer = random.randint(0, 1)
            n = 1 if dealer == 0 else 0
            # Pass init state to players
            d_s2 = env.get_state(dealer)[0]
            n_s2 = env.get_state(n)[0]

            # Setting avg to false, will be overwritten later on
            d_is_avg_strat = False
            n_is_avg_strat = False
            n_t = False
            d_t = False
            terminated = False
            first_round = True
            # print("Reseted env, d_t: {}".format(d_t))
            while not terminated:

                actual_round = env.round_index
                # DEALER
                # If Dealer has not terminated so far do act and env.step
                if not d_t:
                    d_a, d_is_avg_strat = players[dealer].act(d_s2)
                    env.step(d_a, dealer)

                # NOT DEALER
                # N get's new state
                n_s = n_s2
                n_s2, n_a, n_r, n_t = env.get_state(n)
                # Don't memorize experience of first round, because it's unbound to
                # Dealers action
                if not first_round:
                    players[n].remember_by_strategy(n_s, n_a, n_r, n_s2, n_t, n_is_avg_strat)
                # If not terminated so far do act and env.step
                if not n_t:
                    n_a, n_is_avg_strat = players[n].act(n_s2)
                    env.step(n_a, n)

                # DEALER
                # Dealer get's new State after N played as well
                d_s = d_s2
                d_s2, d_a, d_r, d_t = env.get_state(dealer)
                # If it's still same round as at the beginning of the loop, it seems that this round or the
                # game has not terminated -> Dealer has to do an action.
                players[dealer].remember_by_strategy(d_s, d_a, d_r, d_s2, d_t, d_is_avg_strat)
                if actual_round == env.round_index and d_t is False:
                    d_a, d_is_avg_strat = players[dealer].act(d_s2)
                    env.step(d_a, dealer)
                    d_s = d_s2
                    d_s2, d_a, d_r, d_t = env.get_state(dealer)
                    players[dealer].remember_by_strategy(d_s, d_a, d_r, d_s2, d_t, d_is_avg_strat)

                first_round = False

                if n_t and d_t:
                    # Just check if rewards are fitting the rules and the idea of a
                    # zero sum game
                    if abs(d_r) - abs(n_r) != 0 or d_r + n_r > 0 or d_r + n_r < 0:
                        pass
                        # log.debug("="*30)
                        # log.debug("Round: {}".format(actual_round))
                        # log.debug("Dealer is: {}, Dealer-Reward: {}, n-reward: {}".format(dealer, d_r, n_r))
                        # log.debug("Dealer-Action: {}, n-Action:{}".format(np.argmax(d_a), np.argmax(n_a)))
                    terminated = True

        player1.update_strategy()
        player2.update_strategy()


def main(args):

    with tf.Session() as sess:

        env = leduc.Env()
        np.random.seed(int(Config.get('Utils', 'Seed')))
        tf.set_random_seed(int(Config.get('Utils', 'Seed')))

        # initialize dimensions:
        state_dim = env.observation_space
        action_dim = env.action_space

        # initialize players
        player1 = agent.Agent(sess, state_dim, action_dim, 'player1')
        player2 = agent.Agent(sess, state_dim, action_dim, 'player2')

        # Start fictitious self play algorithm
        fsp(sess, env, args, player1, player2)


if __name__ == '__main__':

    print("NFSP by David Joos")
    parser = argparse.ArgumentParser(description='Provide arguments for NFSP agent.')

    # parser.add_argument('--testfunc', help='starts testfunc function instead of main', action='store_true')

    args = vars(parser.parse_args())

    # if args['testfunc'] is True:
    #     pass
    # else:

    main(args)
