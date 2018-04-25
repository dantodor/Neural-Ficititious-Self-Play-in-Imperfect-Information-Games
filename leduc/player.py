from __future__ import print_function

class Player:
    def __init__(self):
        pass

    def act(self, p_card, pot, player_number):
        while True:
            k = raw_input('[1-fold][2-raise][3-call]> ')
            if k == '1':
                return self._fold(pot)
            elif k == '2':
                return self._raise_pot(pot, player_number)
            elif k == '3':
                return self._call(pot, player_number)
            else:
                print ("[Please type in a number between 1 - 3]")

    def set_private_card(self, card):
        self._private_card = card

    def get_private_card(self):
        return self._private_card

    def _fold(self, pot):
        return [False, pot, True]

    def _raise_pot(self, pot, player_number):
        if player_number == 1:
            if pot[0] < pot[1]:
                pot[0] += 2
            else:
                pot[0] += 1
            return [False, pot, False]
        else:
            if pot[0] > pot[1]:
                pot[1] += 2
            else:
                pot[1] += 1
            return [False, pot, False]

    def _call(self, pot, player_number):
        if player_number == 1:
            if pot[0] < pot[1]:
                pot[0] += 1
            return [True, pot, False]
        else:
            if pot[0] > pot[1]:
                pot[1] += 1
            return [True, pot, False]
