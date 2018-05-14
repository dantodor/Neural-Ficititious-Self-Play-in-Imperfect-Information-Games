from __future__ import print_function
import deck
import numpy as np
import ConfigParser
import time


class Env:
    """
    Leduc Hold'em environment.
    TODO: Class description
    """

    def __init__(self):

        # Init config
        self.config = ConfigParser.ConfigParser()
        self.config.read("./config.ini")

        # Init config variables
        self.player_count = int(self.config.get('Environment', 'Playercount'))
        self.decksize = int(self.config.get('Environment', 'Decksize'))
        self.max_rounds = int(self.config.get('Environment', 'MaxRounds'))
        self.suits = int(self.config.get('Environment', 'Suits'))
        self.max_raises = int(self.config.get('Environment', 'MaxRaises'))
        self._action_space = int(self.config.get('Environment', 'ActionSpace'))
        self.total_action_space = int(self.config.get('Environment', 'TotalActionSpace'))

        # Init deck
        self.deck = deck.Deck(self.decksize)

        # Init cards vector
        self.cards = []

        # Who is dealer? Important for blinds.
        self.dealer = 0

        # Init game variables
        self.specific_cards = np.zeros((self.player_count, self.max_rounds, (self.decksize / self.suits)))
        self.round = 0
        self.terminated = False
        self.raises = []
        self.public_card_index = 0
        self.overall_raises = []
        self.reward = np.zeros(self.player_count)
        self.round_raises = 0
        self.last_action = []

        self.actions_done = []
        self.reward_made_index = 0

        # Init specific state
        self.history = np.zeros((self.player_count, self.max_rounds, self.max_raises, self._action_space))

        self.s = np.array([[np.zeros(30)], [np.zeros(30)]])
        # DEBUG
        # self.test = 0

    @property
    def round_index(self):
        return self.round

    @property
    def action_space(self):
        a = np.zeros(self.total_action_space)
        return a.shape

    @property
    def observation_space(self):
        o = self.history.flatten()
        c = self.specific_cards[0].flatten()
        s = np.concatenate((o, c), axis=0)
        s = np.zeros((1, len(s)))
        return s.shape

    def reset(self, dealer):
        self.dealer = dealer
        n = 1 if dealer == 0 else 0
        # Dealer has to set small blind, n_dealer has to set big blind
        self.overall_raises = np.zeros(self.player_count)
        self.overall_raises[dealer] += 0.5
        self.overall_raises[n] += 1

        # Re-init deck as Object of type deck
        self.deck = deck.Deck(self.decksize)
        self.deck.shuffle()

        self.s = np.array([[np.zeros(30)], [np.zeros(30)]])

        self.reward_made_index = 0

        # Reset history to empty
        self.history = np.zeros((self.player_count, self.max_rounds, self.max_raises, self._action_space))

        self.round = 0
        self.terminated = False
        self.raises = np.zeros(self.player_count)
        self.public_card_index = 0
        self.reward = np.zeros(self.player_count)
        self.round_raises = 0
        self.last_action = np.zeros((self.player_count, self.total_action_space))

        self.actions_done = []

        # Init specific cards
        self.specific_cards = np.zeros((self.player_count, self.max_rounds, (self.decksize / self.suits)))

        # Init players specific state
        for k in range(self.player_count):
            card_index = self.deck.pick_up().rank
            # Because 6 cards with 3 duplicates we just need 3 entrys in cards vector
            # Each possible rank is represented by one vector object
            # Set cards vector entry to 1 where the picked up card matches
            self.specific_cards[k][self.round][card_index] = 1

    def get_state(self, p_index):
        cards = self.specific_cards[p_index]
        state = np.concatenate((self.history.flatten(), self.specific_cards[p_index].flatten()))
        action = self.last_action[p_index]

        # Reshape to (1, 1, 3)
        state = np.reshape(state, (1, 1, 30))
        action = np.reshape(action, (1, 1, 3))

        if self.terminated:
            return self.s[p_index], action, self.reward[p_index], state, self.terminated
        else:
            # Return zero as reward because there is no
            return self.s[p_index], action, 0, state, self.terminated

    def do_action(self, action, p_index):

        # Get action with highest value
        # print("THE ACTION: {}".format(action))
        action_value = np.argmax(action)
        self.last_action[p_index] = action

        # If player has raised in this round before - action_value is set to
        # call - 0 to 2 raises are allowed. Maximum one raise per player.
        # AND prevent: Call, Raise, Raise:
        if self.raises[p_index] > 0 and action_value == 2:
            action_value = 1
        if len(self.actions_done) == 2 and self.actions_done[0] == 'Call' and self.actions_done[1] == 'Raise' \
                and action_value == 2:
            action_value = 1

        # Execute actions:
        # Fold:
        if action_value == 0:
            self.actions_done.append('Fold')
            return True

        # Check, call
        elif action_value == 1:
            self.history[p_index][self.round][self.round_raises][0] = 1
            self.round_raises += 1
            if len(self.actions_done) > 0 and self.actions_done[len(self.actions_done) - 1] == "Raise":
                self.overall_raises[p_index] += 1
            if self.round == 0 and len(self.actions_done) == 0:
                # It's the Dealer, he has to double his small blind
                self.overall_raises[p_index] += 0.5
            self.actions_done.append('Call')
            return False

        # Raise
        elif action_value == 2:
            self.history[p_index][self.round][self.round_raises][1] = 1
            self.raises[p_index] += 1
            if len(self.actions_done) > 0 and self.actions_done[len(self.actions_done) - 1] == "Raise":
                self.overall_raises[p_index] += 2
            else:
                self.overall_raises[p_index] += 1
            self.round_raises += 1
            if self.round == 0 and len(self.actions_done) == 0:
                # It's the Dealer, he has to double his small blind
                self.overall_raises[p_index] += 0.5
            self.actions_done.append('Raise')
            return False

    def game_or_round_has_terminated(self):
        if len(self.actions_done) == 2:
            if self.actions_done[0] == 'Call' and self.actions_done[1] == 'Call' \
                    or self.actions_done[0] == 'Raise' and self.actions_done[1] == 'Call':
                return True
        elif len(self.actions_done) == 3:
            if self.actions_done[0] == 'Call' and self.actions_done[1] == 'Raise' and self.actions_done[2] == 'Call' \
                    or self.actions_done[0] == 'Raise' and self.actions_done[1] == 'Raise' and self.actions_done[2] == 'Call':
                return True
        else:
            return False

    def step(self, action, p_index):
        """

        :param action:
        :param p_index:
        :return:
        """

        cards = self.specific_cards[p_index]
        state = np.concatenate((self.history.flatten(), self.specific_cards[p_index].flatten()))
        self.s[p_index][0][:] = state

        if not self.terminated:
            # Deconstruct raises, calls, round etc
            o_index = 1 if p_index == 0 else 0  # TODO: make it dynamic for more than 2 players!

            # Do action
            self.terminated = self.do_action(action, p_index)

            if not self.terminated and self.game_or_round_has_terminated():
                if self.round == 1:
                    # Game has terminated
                    self.terminated = True
                elif self.round == 0:
                    # Update card vector of next round with private card from round_k-1
                    self.specific_cards[p_index][(self.round + 1)] = self.specific_cards[p_index][self.round]
                    self.specific_cards[o_index][(self.round + 1)] = self.specific_cards[o_index][self.round]

                    # Update card vector for next round with revealed public card
                    self.public_card_index = self.deck.pick_up().rank
                    self.specific_cards[p_index][(self.round + 1)][self.public_card_index] = 1
                    self.specific_cards[o_index][(self.round + 1)][self.public_card_index] = 1
                    # print("This the new state:")
                    # print("Player{} - {}".format(p_index, self.specific_cards[p_index][1]))
                    # print("Player{} - {}".format(o_index, self.specific_cards[o_index][1]))

                    # Determine reward from first round
                    # if p_index == self.dealer:
                    #     self.reward[p_index] = self.overall_raises
                    #
                    # self.reward[o_index] = np.sum(self.overall_raises)
                    self.round = 1

                    # Set raises and calls to zero - in new round nothing happened
                    # so far
                    self.raises = np.zeros(2)
                    self.round_raises = 0
                    # Print actions done in round:
                    # print("Player{} has finised".format(p_index))
                    # print("Actions done: {}".format(self.actions_done))
                    self.actions_done = []

                else:
                    print("Round not specified.")
                if abs(self.reward[p_index]) - abs(self.reward[o_index]) != 0:
                    print("IF p-index: {}, o_index: {} ".format(self.reward[p_index], self.reward[o_index]))

            # Determine rewards if terminated
            if self.terminated:
                # Player has folded: Reward is at least zero
                if np.argmax(action) == 0:

                    self.reward[p_index] = self.overall_raises[p_index] * (-1)
                    self.reward[o_index] = self.overall_raises[p_index]

                    if abs(self.reward[p_index]) - abs(self.reward[o_index]) != 0:
                        print("p-index: {}, o_index: {}".format(self.reward[p_index], self.reward[o_index]))

                # No one has folded, check winner by card
                elif np.argmax(action) > 0:
                    # Deconstruct cards
                    p_cards = self.specific_cards[p_index][self.round]
                    o_cards = self.specific_cards[o_index][self.round]

                    # Compute total reward
                    self.reward[p_index] = self.overall_raises[o_index]
                    self.reward[o_index] = self.overall_raises[p_index]

                    # Check if one player has same card as public card
                    # If just one index has value 1, the player has same rank
                    # as public card.
                    if np.count_nonzero(p_cards) == 1:
                        # Player wins
                        self.reward[o_index] *= -1
                    elif np.count_nonzero(o_cards) == 1:
                        # Opponent wins
                        self.reward[p_index] *= -1
                    else:
                        # No player has match with public card, remove public card
                        # if round == 1:
                        if self.round == 1:
                            p_cards[self.public_card_index] = 0
                            o_cards[self.public_card_index] = 0
                        if np.argmax(p_cards) < np.argmax(o_cards):
                            # Player wins
                            self.reward[o_index] *= -1
                        elif np.argmax(p_cards) > np.argmax(o_cards):
                            # Opponent wins
                            self.reward[p_index] *= -1
                        elif np.argmax(p_cards) == np.argmax(o_cards):
                            # Draw
                            self.reward = np.zeros(2)
                        else:
                            print("SOMETHING ELSE HAPPENED"*4)
                        if self.round == 1:
                            p_cards[self.public_card_index] = 1
                            o_cards[self.public_card_index] = 1
                # DEBUG
                # self.test += 1
                # print("===== TEST {} - ROUND {} =====".format(self.test, self.round))
                # print("{}".format(self.actions_done))
                # print("{}".format(self.history[p_index]))
                # print("{}".format(self.history[o_index]))
                # print("REWARD => {}".format(abs(self.reward[p_index])))
                # time.sleep(1)
                # DEBUG
                # cards_0 = self.specific_cards[p_index][self.round]
                # cards_1 = self.specific_cards[o_index][self.round]
                # winner = ''
                # if self.reward[p_index] > self.reward[o_index]:
                #     winner = 'Player' + str(p_index)
                # elif self.reward[p_index] < self.reward[o_index]:
                #     winner = 'Player' + str(o_index)
                # else:
                #     winner = 'Draw'
                #
                # cards_0_string = []
                # for k in range(len(cards_0)):
                #     if cards_0[k] == 1:
                #         if k == 0:
                #             cards_0_string.append('Ace')
                #         elif k == 1:
                #             cards_0_string.append('King')
                #         elif k == 2:
                #             cards_0_string.append('Queen')
                #
                # cards_1_string = []
                # for k in range(len(cards_1)):
                #     if cards_1[k] == 1:
                #         if k == 0:
                #             cards_1_string.append('Ace')
                #         elif k == 1:
                #             cards_1_string.append('King')
                #         elif k == 2:
                #             cards_1_string.append('Queen')
                #
                # print("="*30)
                # print("Winner {} with {} and o_cards: {}".format(winner, cards_0_string, cards_1_string))
                # print("Last action: {}".format(self.actions_done))
                # time.sleep(1)


                if self.reward[p_index] + self.reward[o_index] != 0:
                    print("FUCK MAN")
        else:
            # pass
            print("Player{} tried to step while self.terminated is {}".format(p_index, self.terminated))
            # time.sleep(5)
