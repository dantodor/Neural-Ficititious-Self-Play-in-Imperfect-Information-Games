from __future__ import print_function
import logging
import deck
import numpy as np
import ConfigParser


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
        self.player_count = int(self.config.get('Environment', 'Decksize'))
        self.decksize = int(self.config.get('Environment', 'Decksize'))

        # Init deck
        self.deck = deck.Deck(self.decksize)

        # Init cards vector
        self.cards = []

        # Init game variables
        self.specific_cards = []
        self.round = 0
        self.terminated = False
        self.raises = []
        self.calls = []
        self.public_card = 0

        # Init specific state
        self.history = []

        # Init players vector
        self.players = np.array([])

    def reset(self):
        # Re-init deck as Object of type deck
        self.deck = deck.Deck(self.decksize)
        self.deck.shuffle()

        # Reset history to empy
        self.history = []

        self.round = np.zeros(2)
        self.terminated = False
        self.raises = np.zeros(2)
        self.calls = np.zeros(2)
        self.public_card = 0

        # Init specific cards
        s_cards = np.zeros((self.player_count, (self.decksize / 2)))
        self.specific_cards = np.array([[s_cards], [s_cards]])  # TODO: Make it dynamic

        # Set betting history to 0 because nothing happened so far
        # TODO: Number of zeros depending on config
        players = np.array([])
        rounds = np.zeros(2)
        raises = np.zeros(3)
        actions = np.zeros(2)

        # Init players specific state
        for k in range(self.player_count):
            card_index = self.deck.pick_up().rank
            # Because 6 cards with 3 duplicates we just need 3 entrys in cards vector
            # Each possible rank is represented by one vector object
            # Set cards vector entry to 1 where the picked up card matches
            self.specific_cards[k][0][card_index] = 1

            # Append players index for each player
            self.players = np.append(self.players, [k])

        players_ = np.array([], dtype=object)
        rounds_ = np.array([], dtype=object)
        raises_ = np.array([], dtype=object)

        for raise_ in raises:
            raise_ = np.array([raise_, actions], dtype=object)
            raises_ = np.append(raises_, raise_)

        for round_ in rounds:
            round_ = np.array([round_, raises_], dtype=object)
            rounds_ = np.append(rounds_, round_)

        for player_ in players:
            player_ = np.array([player_, rounds_], dtype=object)
            players_ = np.append(players_, player_)

        self.history = np.array([players_])

    def get_state(self, p_index):
        state = np.array([])

        for player_ in self.history:
            for round_ in player_:
                if type(round_) is not np.int64:
                    for raise_ in round_:
                        if type(raise_) is not np.float64:
                            for action_ in raise_:
                                if type(action_) is not np.float64:
                                    state = np.append(state, action_)

        cards = self.specific_cards[p_index]
        state = np.concatenate((state.flatten(), cards.flatten()))

        return state

    def step(self, action, p_index):
        """

        :param action:
        :param p_index:
        :return:
        """

        # Deconstruct raises and round
        raises = self.raises[p_index]
        round = self.round[p_index]
        o_index = 1 if p_index == 0 else 1

        # Get action with highest value
        action_value = np.argmax(action)
        # For setting action to history
        action_index = action_value - 1

        # Execute actions:
        # Fold:
        if action_value == 0:
            self.terminated = True

        # Check, call
        elif action_value == 1:
            self.history[p_index][round][raises][action_index] = 1

        # Raise
        elif action_value == 2:
            self.history[p_index][round][raises][action_index] = 1
            self.raises[p_index] += 1

        if self.calls[p_index] != 0 and self.raises[p_index] != 0 and self.round == 0 and self.terminated is not False:
            # TODO
            public_card_index = self.deck.pick_up().rank
            self.specific_cards[p_index][1][public_card_index] = 1
            self.specific_cards[o_index][1][public_card_index] = 1





        pass

