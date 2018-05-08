import sys, logging
import tensorflow as tf
import leduc.newenv as leduc
import agent.agent as agent
import leduc.human as human
import numpy as np
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

    players = [player1, player2]
    eta = float(Config.get('Agent', 'Eta'))
    # Randomly set dealer
    dealer = random.randint(0, 1)
    n = 1 if dealer == 0 else 0


    for i in range(int(Config.get('Common', 'MaxEpisodes'))):
        # Exploitability
        R1 = 0
        R2 = 0

        steps_0 = 0
        steps_1 = 0
        actions = np.zeros((2, 3))
        for j in range(int(Config.get('Common', 'Episodes'))):

            env.reset()

            # Toggle dealer
            if dealer == 0:
                dealer = 1
                n = 0
            else:
                dealer = 0
                n = 1
            reward_0 = 0
            reward_1 = 0

            # Pass init state to players
            d_s2 = env.get_state(dealer)[0]
            n_s2 = env.get_state(n)[0]

            d_t = False

            terminated = False
            first_round = True
            while not terminated:

                actual_round = env.round_index
                # DEALER
                # If Dealer has not terminated so far do act and env.step
                if not d_t:
                    # Predict best and average response
                    d_a_br = players[dealer].act_best_response(d_s2)
                    d_a_avgr = players[dealer].act_average_response(d_s2)
                    # Compute convex combination of predictions
                    d_a_mix = players[dealer].mixed_strategy(d_a_br, d_a_avgr)
                    if np.random.rand(1) > eta:
                        env.step(d_a_avgr, dealer)
                    else:
                        env.step(d_a_br, dealer)
                        players[dealer].remember_best_response(d_s2, d_a_br)
                    steps_0 += 1 if dealer == 0 else 0
                    steps_1 += 1 if dealer == 1 else 0

                    if not first_round:
                        actions[dealer][np.argmax(d_a_mix)] += 1

                # NOT DEALER
                # N get's new state
                n_s = n_s2
                n_s2, n_a, n_r, n_t = env.get_state(n)
                # Don't memorize experience of first round, because it's unbound to
                # Dealers action
                if not first_round:
                    players[n].remember_for_rl(n_s, n_a, n_r, n_s2, n_t)
                # If not terminated so far do act and env.step
                if not n_t:
                    n_a_br = players[n].act_best_response(n_s2)
                    n_a_avgr = players[n].act_average_response(n_s2)
                    n_a_mix = players[n].mixed_strategy(n_a_br, n_a_avgr)
                    if np.random.rand(1) > eta:
                        env.step(n_a_avgr, n)
                    else:
                        env.step(n_a_br, n)
                        players[n].remember_best_response(n_s2, n_a_br)
                    steps_0 += 1 if n == 0 else 0
                    steps_1 += 1 if n == 1 else 0
                    actions[n][np.argmax(n_a_mix)] += 1

                # DEALER
                # Dealer get's new State after N played as well
                d_s = d_s2
                d_s2, d_a, d_r, d_t = env.get_state(dealer)
                # If it's still same round as at the beginning of the loop, it seems that this round or the
                # game has not terminated -> Dealer has to do an action.
                players[dealer].remember_for_rl(d_s, d_a, d_r, d_s2, d_t)
                if actual_round == env.round_index and d_t is False:
                    # Predict best and average response
                    d_a_br = players[dealer].act_best_response(d_s2)
                    d_a_avgr = players[dealer].act_average_response(d_s2)
                    # Compute convex combination of predictions
                    d_a_mix = players[dealer].mixed_strategy(d_a_br, d_a_avgr)
                    if np.random.rand(1) > eta:
                        env.step(d_a_avgr, dealer)
                    else:
                        env.step(d_a_br, dealer)
                        players[dealer].remember_best_response(d_s2, d_a_br)
                    actions[dealer][np.argmax(d_a_mix)] += 1

                    steps_0 += 1 if dealer == 0 else 0
                    steps_1 += 1 if dealer == 1 else 0
                    d_s = d_s2
                    d_s2, d_a, d_r, d_t = env.get_state(dealer)
                    players[dealer].remember_for_rl(d_s, d_a, d_r, d_s2, d_t)

                first_round = False

                if n_t and d_t:
                    # Just check if rewards are fitting the rules and the idea of a
                    # zero sum game
                    if abs(d_r) - abs(n_r) != 0 or d_r + n_r > 0 or d_r + n_r < 0:

                        log.debug("="*30)
                        log.debug("Round: {}".format(actual_round))
                        log.debug("Dealer is: {}, Dealer-Reward: {}, n-reward: {}".format(dealer, d_r, n_r))
                        log.debug("Dealer-Action: {}, n-Action:{}".format(np.argmax(d_a), np.argmax(n_a)))
                    terminated = True
                    reward_0 += n_r if n == 0 else d_r
                    reward_1 += n_r if n == 1 else d_r

        for player in players:
            player.update_strategy()

        # measure exploitability
        # print("Begin Meassuring...")
        # counter = 0
        # term = False
        # while not term:
        #     counter += 1
        #     if counter > 300:
        #         term = True
        #     env.reset()
        #     s = env.get_state(0)[0]
        #     a = players[0].act_average_response(s)
        #     env.step(a, 0)
        #     s, a, r, t = env.get_state(1)
        #     if not t:
        #         a = players[1].act_best_response(s)
        #         env.step(a, 1)
        #         s, a, r, t = env.get_state(0)
        #         if not t:
        #             a = players[0].act_average_response(s)
        #             R1 = np.max(a)
        #             term = True
        # if counter > 300:
        #     print("Not enough computational budget. Skipping meassuring")
        #     while not term:
        #         env.reset()
        #         s = env.get_state(1)[0]
        #         a = players[0].act_average_response(s)
        #         env.step(a, 1)
        #         s, a, r, t = env.get_state(0)
        #         if not t:
        #             a = players[0].act_best_response(s)
        #             R2 = np.max(a)
        #             term = True
        # print("Meassuring endet.")

        # print("======== Exploitability of {} ========".format(R1+R2))
        # print("Player0 got {} reward with Folds: {}, Calls: {}, Raises: {}".format(reward_0, actions[0][0], actions[0][1], actions[0][2]))
        # print(
        #     "Player1 got {} reward with Folds: {}, Calls: {}, Raises: {}".format(reward_1, actions[1][0], actions[1][1],
        #                                                                          actions[1][2]))
        reward_0 = 0
        reward_1 = 0


def human_interaction(sess, env, args, player1):
    player2 = human.Human("Dave")
    players = [player1, player2]
    for i in range(int(Config.get('Common', 'MaxEpisodes'))):
        steps_0 = 0
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
            first_round = True
            terminated = False
            while not terminated:
                actual_round = env.round_index
                # DEALER
                # If Dealer has not terminated so far do act and env.step
                if not d_t:
                    if dealer == 0:
                        d_a, d_is_avg_strat = players[dealer].act(d_s2)
                        env.step(d_a, dealer)
                    else:
                        players[dealer].show_state(env.get_state(dealer)[0], actual_round)
                        env.step(players[dealer].act(), dealer)

                # NOT DEALER
                # N get's new state
                n_s = n_s2
                n_s2, n_a, n_r, n_t = env.get_state(n)
                # Don't memorize experience of first round, because it's unbound to
                # Dealers action
                if not first_round and n == 0:
                    players[n].remember_by_strategy(n_s, n_a, n_r, n_s2, n_t, n_is_avg_strat)
                # If not terminated so far do act and env.step
                if not n_t:
                    if n == 0:
                        n_a, n_is_avg_strat = players[n].act(n_s2)
                        env.step(n_a, n)
                    else:
                        players[n].show_state(env.get_state(n)[0], actual_round)
                        env.step(players[n].act(), n)

                # DEALER
                # Dealer get's new State after N played as well
                d_s = d_s2
                d_s2, d_a, d_r, d_t = env.get_state(dealer)
                # If it's still same round as at the beginning of the loop, it seems that this round or the
                # game has not terminated -> Dealer has to do an action.
                if dealer == 0:
                    players[dealer].remember_by_strategy(d_s, d_a, d_r, d_s2, d_t, d_is_avg_strat)
                if actual_round == env.round_index and d_t is False:
                    if dealer == 0:
                        d_a, d_is_avg_strat = players[dealer].act(d_s2)
                        env.step(d_a, dealer)
                        d_s = d_s2
                        d_s2, d_a, d_r, d_t = env.get_state(dealer)
                        players[dealer].remember_by_strategy(d_s, d_a, d_r, d_s2, d_t, d_is_avg_strat)
                    else:
                        print("LAST ACTION!")
                        players[dealer].show_state(d_s2, actual_round)
                        if not d_t:
                            env.step(players[dealer].act(), dealer)
                            d_s2, d_a, d_r, d_t = env.get_state(dealer)

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

                    if dealer == 1:
                        print"Dealer was Human"
                        op_cards = n_s2[0][24:30]
                        players[dealer].show_winner(d_r, op_cards, n_a)
                    else:
                        print("Dealer was Agent")
                        op_cards = d_s2[0][24:30]
                        players[n].show_winner(n_r, op_cards, d_a)

                if steps_0 != 0 and steps_0 % 128 == 0:
                    players[0].update_strategy()




def main(args):

    with tf.Session() as sess:

        env = leduc.Env()
        # np.random.seed(int(Config.get('Utils', 'Seed')))
        tf.set_random_seed(int(Config.get('Utils', 'Seed')))

        # initialize dimensions:
        state_dim = env.observation_space
        action_dim = env.action_space

        # initialize players
        player1 = agent.Agent(sess, state_dim, action_dim, 'Player0')
        player2 = agent.Agent(sess, state_dim, action_dim, 'Player1')

        # initialize tensorflow variables
        sess.run(tf.global_variables_initializer())

        # Start fictitious self play algorithm
        if args['human'] is True:
            human_interaction(sess, env, args, player1)
        else:
            fsp(sess, env, args, player1, player2)


if __name__ == '__main__':

    print("NFSP by David Joos")
    parser = argparse.ArgumentParser(description='Provide arguments for NFSP agent.')

    parser.add_argument('--human', help='starts testfunc function instead of main', action='store_true')

    args = vars(parser.parse_args())

    main(args)
