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
        Resets the environment. Status of the game is then: initialized.

        :return: None
        """
        self._deck = deck.Deck(self._decksize)
        self._deck.shuffle()
        self._public_card = self._deck.fake_pub_card()
        self._pot = [0, 0]
        self.pot = 0

        for _ in range(self.player_count):
            # Define return tuple for step()
            card = self._deck.pick_up().rank
            return_tuple = np.array([[
                [card, self._public_card.rank, self.pot],   # state
                self._action_space,                         # action
                self._reward,                               # reward
                [card, self._public_card.rank, self.pot],   # next state
                self._terminal,                             # terminal
                self._info                                  # info
            ]])
            # Append tuple for each player which got a different card from deck
            self._specific_state.append(return_tuple)

    def init_state(self, player_index):
        """
        Provides a initial state for the players. Because of the imperfect information state
        every player get's his own initial state.

        IMPORTANT: s and s2 are the same in this case. Don't use it for learning. No
        transition has been happened to this point.

        :param player_index:
        :return: inital_state -> s, a, r, s, t, i
        """
        return self._specific_state[player_index]

    def step(self, action, player_index):
        """
        Does exactly one step depending on given action.

        :param action: np.array((1,3))
        :param player_index: Index of player = 0 OR 1
        :return: s, a, r, s2, t, i (s = state, a = action, r = reward, s2 = new state, t = terminated, i = info)

        action[0]: fold
        action[1]: call
        action[2]: raise
        """

        # Buffer to safe state before the transition to the next state happened
        state_buffer = self._specific_state[player_index][3]

        # Check's if both players finished first betting round. If so, reveal the public card to state
        if self._left_choices[player_index] <= 2 or self._left_choices[1 if player_index == 0 else 0] <= 2:
            if self._public_card.rank == 0:
                self._public_card = self._deck.pick_up()
                self._specific_state[player_index][0][3] = self._public_card.rank
                self._specific_state[1 if player_index == 0 else 0][0][3] = self._public_card.rank

        # Check if player has the right to take action
        if self._left_choices[player_index] > 0:

            # Compares action probabilities against each other. Action with highest probability will be used
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

        # If player has lost his right to take action (no action to take left) or
        # folded: self._terminal will be 1.
        self._specific_state[player_index][4] = self._terminal

        # Evaluates winner
        if self._left_choices[player_index] == 0 and self._left_choices[1 if player_index == 0 else 0] <= 2:
            # Player wins:
            if self._specific_state[player_index][3][0] == self._specific_state[player_index][3][1]:
                self._specific_state[player_index][2] = self._pot[1 if player_index == 0 else 0]
            # Opponent wins:
            elif self._specific_state[1 if player_index == 0 else 0][3][0] == self._specific_state[player_index][3][1]:
                self._specific_state[player_index][2] = self._pot[player_index] * (-1)
            # Opponent wins:
            elif self._specific_state[player_index][3][0] > self._specific_state[1 if player_index == 0 else 0][3][0]:
                self._specific_state[player_index][2] = self._pot[player_index] * (-1)
            # Player wins:
            elif self._specific_state[player_index][3][0] < self._specific_state[1 if player_index == 0 else 0][3][0]:
                self._specific_state[player_index][2] = self._pot[1 if player_index == 0 else 0]
            # Draw
            else:
                self._specific_state[player_index][2] = 0

        # Computes pot by an addition of each players specific pot
        self.pot = self._pot[player_index] + self._pot[1 if player_index == 0 else 0]
        # Updates pot
        self._specific_state[player_index][3][2] = self.pot
        # Now state_buffer is the old state:
        self._specific_state[player_index][0] = state_buffer
        # Store the action
        self._specific_state[player_index][1] = action
        # Set terminal to 0 - The other player may has some actions to take left.
        self._terminal = 0

        # Return state, action, reward, next state, terminal, info
        return self._specific_state[player_index]
