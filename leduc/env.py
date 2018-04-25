from __future__ import print_function
import logging
import deck
import numpy as np
import ConfigParser


class Env:
    """Game interface which handles rules, process and players"""

    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        self.config.read("./config.ini")
        self.player_count = int(self.config.get('Environment', 'Playercount'))
        self._decksize = int(self.config.get('Environment', 'Decksize'))
        self._terminal = 0
        self._pot = [0, 0]
        self.pot = 0
        self.state = []
        self._deck = deck.Deck(self._decksize)
        self._public_card = []
        self._specific_state = []
        self.info = ""
        self._left_choices = [int(self.config.get('Environment', 'Choices')), int(self.config.get('Environment', 'Choices'))]
        self._observation_state = np.zeros((1, 3))
        self._action_space = np.zeros((1, 3))

    @property
    def observation_space(self):
        return self._observation_state.shape

    @property
    def action_space(self):
        return self._action_space.shape

    def reset(self):
        """
        :return: null
        """
        self._deck = deck.Deck(self._decksize)
        self._deck.shuffle()
        self._public_card = self._deck.fake_pub_card()
        self._pot = [0, 0]
        self.pot = 0
        for _ in range(self.player_count):
            card = self._deck.pick_up()
            self._specific_state.append([[card.rank(), self._public_card.rank(), self.pot],
                                         0, self._terminal, self.info])


    def init_state(self, player_index):
        return self._specific_state[player_index]

    def step(self, action, player_index):
        """
        :param action: np.array((1,3))
        :param player_index: Index of player = 0 OR 1
        :return: state, reward, terminal, info

        Does exactly one step depending on given action.
        action[0]: fold
        action[1]: call
        action[2]: raise
        """

        if self._left_choices[player_index] <= 2 or self._left_choices[1 if player_index == 0 else 0] <= 2:
            if self._public_card.rank() == 0:
                self._public_card = self._deck.pick_up()
                self._specific_state[player_index][0][1] = self._public_card.rank()
                self._specific_state[1 if player_index == 0 else 0][0][1] = self._public_card.rank()

        if self._left_choices[player_index] > 0:
            if action[0] > action[1] and action[0] > action[2]:
                # fold -> terminate, shift reward to opponent
                print("FOLDING")
                self._terminal = 1

            if action[1] > action[0] and action[1] > action[2]:
                print("CALLING")
                # call -> fit pot
                self._left_choices[player_index] -= 1
                if self._pot[player_index] < self._pot[1 if player_index == 0 else 0]:
                    self._pot[player_index] += 1
                self._terminal = 1 if self._left_choices[player_index] == 0 else 0

            if action[2] > action[0] and action[2] > action[1]:
                print("RAISING")
                # raise
                self._left_choices[player_index] -= 2
                if self._pot[player_index] <= self._pot[1 if player_index == 0 else 0]:
                    self._pot[player_index] += 1
                self._terminal = 1 if self._left_choices[player_index] == 0 else 0
        else:
            self._terminal = 1

        self._specific_state[player_index][2] = self._terminal

        if self._left_choices[player_index] == 0 and self._left_choices[1 if player_index == 0 else 0] <= 2:
            # player wins:
            if self._specific_state[player_index][0][0] == self._specific_state[player_index][0][1]:
                self._specific_state[player_index][1] = self._pot[1 if player_index == 0 else 0]
            # opponent wins:
            elif self._specific_state[1 if player_index == 0 else 0][0][0] == self._specific_state[player_index][0][1]:
                self._specific_state[player_index][1] = self._pot[player_index] * (-1)
            # opponent wins:
            elif self._specific_state[player_index][0][0] > self._specific_state[1 if player_index == 0 else 0][0][0]:
                self._specific_state[player_index][1] = self._pot[player_index] * (-1)
            # player wins:
            elif self._specific_state[player_index][0][0] < self._specific_state[1 if player_index == 0 else 0][0][0]:
                self._specific_state[player_index][1] = self._pot[1 if player_index == 0 else 0]
            # draw
            else:
                self._specific_state[player_index][1] = 0

        self.pot = self._pot[player_index] + self._pot[1 if player_index == 0 else 0]
        self._specific_state[player_index][0][2] = self.pot
        self._terminal = 0

        return self._specific_state[player_index]
