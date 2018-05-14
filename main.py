import sys, logging
import tensorflow as tf
import leduc.newenv as leduc
import agent.agent as agent
import numpy as np
import argparse
import ConfigParser
import random
import logging
import time
import matplotlib.pyplot as plt
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
log = logging.getLogger('')


# Configuration
Config = ConfigParser.ConfigParser()
Config.read("./config.ini")


def train(env, player1, player2):
    eta = float(Config.get('Agent', 'Eta'))
    players = [player1, player2]
    dealer = random.randint(0, 1)
    plotter = []

    for i in range(int(Config.get('Common', 'Episodes'))):
        if dealer == 0:
            dealer = 1
        else:
            dealer = 0
        # Set dealer, reset env and pass dealer to it
        env.reset(dealer)

        lhand = 1 if dealer == 0 else 0
        policy = np.array(['', ''])
        # Set policies sigma
        if random.random() > eta:
            policy[dealer] = 'a'
        else:
            policy[dealer] = 'b'
        if random.random() > eta:
            policy[lhand] = 'a'
        else:
            policy[lhand] = 'b'

        # Observe initial state for dealer
        d_s = env.get_state(dealer)[3]

        terminated = False
        first_round = True
        d_t = False
        l_t = False

        while not terminated:
            actual_round = env.round_index
            if first_round and not d_t:
                d_t = players[dealer].play(policy[dealer], dealer, d_s)
                first_round = False
            elif not first_round and not d_t:
                d_t = players[dealer].play(policy[dealer], dealer)
            if not l_t:
                l_t = players[lhand].play(policy[lhand], lhand)
            if actual_round == env.round_index and not d_t:
                d_t = players[dealer].play(policy[dealer], dealer)
            if d_t and l_t:
                terminated = True

        if i > 150 and i % 100 == 0:
            print("================ Stats ==================")
            for player in players:
                player.sampled_actions()
            ex = players[0].average_payoff_br() + players[1].average_payoff_br()
            print("Exploitability: {}".format(ex))
            plotter.append(ex)

        # if i % 50 == 0:
        #     print("================ Stats ==================")
        #     for player in players:
        #         player.sampled_actions()

        # Measure exploitability
        # if i % 20 == 0:
        #     expl = []
        #     dealer = random.randint(0, 1)
        #     for i in range(int(Config.get('Common', 'TestEpisodes'))):
        #         if dealer == 0:
        #             dealer = 1
        #         else:
        #             dealer = 0
        #         for player in players:
        #             player.play_test_init()
        #         env.reset(dealer)
        #         lhand = 1 if dealer == 0 else 0
        #         policy[dealer] = 'b'
        #         policy[lhand] = 'a'
        #         d_s = env.get_state(dealer)[3]
        #
        #         terminated = False
        #         first_round = True
        #         d_t = False
        #         l_t = False
        #
        #         while not terminated:
        #             actual_round = env.round_index
        #             if first_round and not d_t:
        #                 d_t = players[dealer].play_test(policy[dealer], dealer, d_s)
        #                 first_round = False
        #             elif not first_round and not d_t:
        #                 d_t = players[dealer].play_test(policy[dealer], dealer)
        #             if not l_t:
        #                 l_t = players[lhand].play_test(policy[lhand], lhand)
        #             if actual_round == env.round_index and not d_t:
        #                 d_t = players[dealer].play_test(policy[dealer], dealer)
        #             if d_t and l_t:
        #                 terminated = True
        #                 a = players[dealer].play_test_get_reward
        #                 expl.append(a)
        #     plotter.append(np.average(expl))
        #     print("==== EXPLOITABILITY: {}".format(np.average(expl)))

    plt.plot(plotter)
    plt.show()
    time.sleep(60)


def main(args):

    with tf.Session() as sess:

        env = leduc.Env()
        np.random.seed(int(Config.get('Utils', 'Seed')))
        tf.set_random_seed(int(Config.get('Utils', 'Seed')))

        # initialize dimensions:
        state_dim = env.observation_space
        action_dim = env.action_space

        # initialize players
        player1 = agent.Agent(sess, state_dim, action_dim, 'Player0', env)
        player2 = agent.Agent(sess, state_dim, action_dim, 'Player1', env)

        # initialize tensorflow variables
        sess.run(tf.global_variables_initializer())

        train(env, player1, player2)


if __name__ == '__main__':

    print("NFSP by David Joos")
    parser = argparse.ArgumentParser(description='Provide arguments for NFSP agent.')

    parser.add_argument('--human', help='starts testfunc function instead of main', action='store_true')

    args = vars(parser.parse_args())

    main(args)
