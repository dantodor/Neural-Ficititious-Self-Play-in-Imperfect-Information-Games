from __future__ import print_function
import logging
import deck
import numpy as np
import ConfigParser


class Env:
    """
    Game environment. Provides functions for the player to interact with the game.
    """

    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        self.config.read("./config.ini")
        self.player_count = int(self.config.get('Environment', 'Playercount'))
        self._decksize = int(self.config.get('Environment', 'Decksize'))
        self._terminal = 0
        self._reward = 0
        self._pot = [0, 0]
        self.pot = 0
        self.state = []
        self._deck = deck.Deck(self._decksize)
        self._public_card = []
        self._specific_state = []
        self._info = ""
        self._left_choices = [int(self.config.get('Environment', 'Choices')), int(self.config.get('Environment', 'Choices'))]

        # dimensions
        self._state_shape = np.zeros((1, 3))
        self._observation_state = 3  #np.zeros((1, 3))
        self._action_space = 3  #np.zeros((1, 3))

    @property
    def dim_shape(self):
        return self._state_shape.shape

    @property
    def observation_space(self):
        return self._observation_state

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        """
        Resets the environment. Status of the game is then: initialized.

        :return: None
        """
        self._deck = deck.Deck(self._decksize)
        self._deck.shuffle()
        self._public_card = self._deck.fake_pub_card()
        self._pot = [0, 0]
        self.pot = 0
        self._specific_state = []
        self._left_choices = [int(self.config.get('Environment', 'Choices')), int(self.config.get('Environment', 'Choices'))]

        for j in range(self.player_count):
            # Define return tuple for step()
            card = self._deck.pick_up().rank
            # print("Player with index {} got card with rank: {}".format(j,card))
            return_tuple = np.array([
                np.array([[card, self._public_card.rank, self.pot]]),   # state
                self._action_space,                         # action
                self._reward,                               # reward
                self._terminal,                             # terminal
                self._info                                  # info
            ])
            # Append tuple for each player which got a different card from deck
            self._specific_state.append(return_tuple)

    def init_state(self, player_index):
        """
        Provides a initial state for the players. Because of the imperfect information state
        every player get's his own initial state.

        :param player_index:
        :return: inital_state -> s as initial_state
        """
        return self._specific_state[player_index][0]

    def step(self, action, player_index):
        """
        Does exactly one step depending on given action.

        :param action: int in range(0, 2)
        :param player_index: Index of player = 0 OR 1

        action = 0: fold
        action = 1: call
        action = 2: raise
        """

        # Check's if both players finished first betting round. If so, reveal the public card to state
        if self._left_choices[player_index] <= 2 or self._left_choices[1 if player_index == 0 else 0] <= 2:
            if self._public_card.rank == 0:
                self._public_card = self._deck.pick_up()
                self._specific_state[player_index][0][0][3] = self._public_card.rank
                self._specific_state[1 if player_index == 0 else 0][0][0][3] = self._public_card.rank

        # Check if player has the right to take action
        if self._left_choices[player_index] > 0:

            # Get index of action with highest value
            action_value = np.argmax(action)
            # if player_index == 0:
            #     print("Player1 has done: {}".format(action_value))
            # else:
            #     print("Player2 has done: {}".format(action_value))

            # Act
            if action_value == 0:
                # fold -> terminate, shift reward to opponent
                # print("FOLDING")
                # Penalty for instant folding even if opponent hasn't raised so far
                if self._left_choices[player_index] == 4 and self._left_choices[1 if player_index == 0 else 0] == 4:
                    self._specific_state[player_index][2] = int(self.config.get('Agent', 'Penalty'))
                self._terminal = 1

            if action_value == 1:
                # print("CALLING")
                # call -> fit pot
                self._left_choices[player_index] -= 1
                if self._pot[player_index] < self._pot[1 if player_index == 0 else 0]:
                    self._pot[player_index] += 1
                self._terminal = 1 if self._left_choices[player_index] == 0 else 0

            if action_value == 2:
                # print("RAISING")
                # raise
                # Penalty for doing a raise but hasn't the right to
                if self._left_choices[player_index] % 2 != 0:
                    self._specific_state[player_index][2] = int(self.config.get('Agent', 'Penalty'))
                self._left_choices[player_index] -= 2
                if self._pot[player_index] <= self._pot[1 if player_index == 0 else 0]:
                    self._pot[player_index] += 1
                self._terminal = 1 if self._left_choices[player_index] == 0 else 0
        else:
            self._terminal = 1

        # If player has lost his right to take action (no action to take left) or
        # folded: self._terminal will be 1.
        self._specific_state[player_index][3] = self._terminal

        # Computes pot by an addition of each players specific pot
        self.pot = self._pot[player_index] + self._pot[1 if player_index == 0 else 0]
        # Updates pot
        self._specific_state[player_index][0][0][2] = self.pot
        # Store the action
        self._specific_state[player_index][1] = action

        # Update terminal state
        self._specific_state[player_index][3] = self._terminal

        # Set terminal to 0 - The other player may has some actions to take left.
        self._terminal = 0

    def get_new_state(self, player_index):
        """
        Returns new state after both players has taken step

        :param player_index: int in range (0, 1)
        :return: s, a, r, t, i (s = state, a = action, r = reward, s2 = new state, t = terminated, i = info)
        """
        # print("This is the new state: {}".format(self._specific_state[player_index][3]))

        # Computes pot by an addition of each players specific pot
        self.pot = self._pot[player_index] + self._pot[1 if player_index == 0 else 0]
        # Updates pot
        self._specific_state[player_index][0][0][2] = self.pot
        # Means, for each player game has terminated if opponent has terminated
        self._specific_state[player_index][3] = self._specific_state[1 if player_index == 0 else 0][3]

        # If game has terminated, winner can be evaluated
        if self._specific_state[player_index][3] == 1:
            # Player wins:
            if self._specific_state[player_index][0][0][0] == self._specific_state[player_index][0][0][1]:
                # if player_index == 0:
                #     print("player1 had same card")
                # elif player_index == 1:
                #     print("player2 had same card")
                self._specific_state[player_index][2] += self._pot[1 if player_index == 0 else 0]
            # Opponent wins:
            elif self._specific_state[1 if player_index == 0 else 0][0][0][0] == self._specific_state[player_index][0][0][1]:
                self._specific_state[player_index][2] += self._pot[player_index] * (-1)
            # Opponent wins:
            elif self._specific_state[player_index][0][0][0] > self._specific_state[1 if player_index == 0 else 0][0][0][0]:
                self._specific_state[player_index][2] += self._pot[player_index] * (-1)
            # Player wins:
            elif self._specific_state[player_index][0][0][0] < self._specific_state[1 if player_index == 0 else 0][0][0][0]:
                # print("Card player1: {}, Card player2: {}".format(self._specific_state[player_index][0][0][0], self._specific_state[1 if player_index == 0 else 0][0][0][0]))
                # print("Complete state: {}".format(self._specific_state[player_index][0]))
                # if player_index == 0:
                #     print("player1 had higher card")
                # elif player_index == 1:
                #     print("player2 had higher card")
                self._specific_state[player_index][2] += self._pot[1 if player_index == 0 else 0]
            # Draw
            else:
                self._specific_state[player_index][2] += 0
        self._terminal = 0

        return self._specific_state[player_index]
