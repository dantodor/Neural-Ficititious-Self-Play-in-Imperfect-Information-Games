
from __future__ import print_function
import logging
import deck
import numpy as np
import ConfigParser

players = np.array([0, 1])
rounds = np.zeros(2)
raises = np.zeros(3)
actions = np.zeros(2)

players_ = np.array([], dtype=object)
rounds_ = np.array([], dtype=object)
raises_ = np.array([], dtype=object)
actions_ = np.array([], dtype=object)

history = np.array([])

for raise_ in raises:
    raise_ = np.array([raise_, actions], dtype=object)
    raises_ = np.append(raises_, raise_)

for round_ in rounds:
    round_ = np.array([round_, raises_], dtype=object)
    rounds_ = np.append(rounds_, round_)

for player_ in players:
    player_ = np.array([player_, rounds_], dtype=object)
    players_ = np.append(players_, player_)

history = np.array([players_])


state = np.array([])

for player_ in history:
    for round_ in player_:
        if type(round_) is not np.int64:
            for raise_ in round_:
                if type(raise_) is not np.float64:
                    for action_ in raise_:
                        if type(action_) is not np.float64:
                            state = np.append(state, action_)

cards = np.array([[1, 2, 3], [4, 5, 6]])

state = np.concatenate((state.flatten(), cards.flatten()))

print(state.flatten())

# print(state)
# counter = 0
#
# p = 0
# r = 1
# ra = 0
#
# print(history[p][r][ra])
