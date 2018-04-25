from __future__ import print_function
import logging
import deck
import numpy as np
import ConfigParser


class LeducEnv:
    """Game interface which handles rules, process and players"""

    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        self.config.read("./config.ini")
        self.player_count = self.config.get('Environment', 'Playercount')
        self._terminal = 0
        self._pot = [0, 0]
        self.pot = 0
        self.state = []
        self._public_card = []
        self._deck = []
        self._specific_state = []
        self.info = ""
        self.reset()

    def reset(self):
        """Initializes game environemt. Shuffles deck, hand out cards."""
        self._deck = deck.Deck(int(self.config.get('Environment', 'Decksize')))
        self._deck.shuffle()
        self._public_card = []
        self._pot = [0, 0]
        self.pot = 0
        for _ in self.player_count:
            self._specific_state.append([self._deck.pick_up(), 0, self.pot, self.terminal, self.info])

    def step(self, action, player_index):
        """
        :param action: np.array((1,3))
        :return: state, reward, terminal, info

        Does exactly one step depending on given action.
        action[0]: fold
        action[1]: call
        action[2]: raise
        """

        if action[0] > action[1] and action[0] > action[2]:
            # fold -> terminate, shift reward to opponent
            self._terminal = 1
            self._specific_state[player_index][1] = self._pot[player_index] * (-1)
            self._specific_state[player_index][3] = self._terminal
            self._specific_state[1 if player_index == 0 else 0][1] = self._pot[player_index]
            self._specific_state[1 if player_index == 0 else 0][3] = self._terminal

        if action[1] > action[0] and action[1] > action[2]:
            # call -> fit pot
            if self._pot[player_index] < self._pot[1 if player_index == 0 else 0]:
                self._pot[player_index] += 1

        if action[2] > action[0] and action[2] > action[1]:
            # raise
            if self._pot[player_index] <= self._pot[1 if player_index == 0 else 0]:
                self._pot[player_index] += 1

        self.pot = self._pot[player_index] + self._pot[1 if player_index == 0 else 0]
        self._specific_state[player_index][2] = self.pot
        return self._specific_state[player_index][0], \
               self._specific_state[player_index][1], \
               self._specific_state[player_index][2], \
               self._specific_state[player_index][3], \
               self._specific_state[player_index][4]

    def _play_betting_round(self):
        """Handles betting rounds."""
        p1_f_b, p2_f_b, p1_fold, p2_fold = self._players_state
        while True:
            if p1_fold == True or p2_fold == True:
                break
            elif p1_f_b == True and p2_f_b == True:
                break
            else:
                p1_f_b, p1_bet, p1_fold = self._player1.act(self._public_card, self._pot, 1)
                self._pot = p1_bet
                p2_f_b, p2_bet, p2_fold = self._player2.act(self._public_card, self._pot, 2)
                self._pot = p2_bet
        return [p1_f_b, p2_f_b, p1_fold, p2_fold]

    def _evaluate_winner(self):
        """Evaluates the winner for a specific round"""
        p1_rank = self._player1.get_private_card().rank
        logging.debug("Player 1: %s" % self._player1.get_private_card().__str__())
        p2_rank = self._player2.get_private_card().rank
        logging.debug("Player 2: %s" % self._player2.get_private_card().__str__())
        pu_rank = self._public_card.rank
        logging.debug("Public Card: %s" % self._public_card.__str__())
        if p1_rank == pu_rank:
            logging.debug("Player 1 wins.")
            # Player 1 wins
        elif p2_rank == pu_rank:
            logging.debug("Player 2 wins.")
            # Player 2 wins
        elif p1_rank == p2_rank:
            logging.debug("Draw.")
            # Draw
        elif p1_rank > p2_rank:
            logging.debug("Player 1 wins.")
            # Player 1 wins
        else:
            logging.debug("Player 2 wins.")
            # Player 2 wins

