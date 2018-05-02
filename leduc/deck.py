from __future__ import print_function
from random import shuffle as rshuffle
import cardmatrix as cm

class Card:
    """A standart playing card with rank and suit."""
    def __init__(self, rank, suit):
        self._rank = rank
        self._suit = suit
        _cm = cm.Cardmatrix()
        self._named_rank, self._named_suit = _cm.getCard(rank, suit)

    def __str__(self):
        return str(self._named_rank) + ' ' \
        + str(self._named_suit) + ' ' \
        + str(self._rank) + ' ' \
        + str(self._suit)
    
    def _print_human_style(self):
        print(str(self._named_suit + ' ' + self._named_rank))

    @property
    def rank(self):
        return self._rank


class Deck:
    """A reduced deck with six cards, 2 suits and 3 ranks."""
    def __init__(self, size=6):
        assert size > 0 and size % 2 == 0, 'Decksize has to be an even number which is greater than 0.'
        self._size = size
        self._fill()
        self.fake_pub = Card(-1, -1)

    def _fill(self):
        """Fill the deck with unshuffeled amount of cards => size."""
        cards_per_suit = self._size / 2  # suits are anyway not necessary for this game.
        self._cards = [Card(rank, suit)
                for rank in range(cards_per_suit)
                for suit in range(2)]
    
    def shuffle(self):
        """Shuffle deck"""
        rshuffle(self._cards)

    def fake_pub_card(self):
        return self.fake_pub

    def pick_up(self):
        return self._cards.pop()

    def print_deck(self):
        """Print each card from deck"""
        for card in self._cards:
            print(card.__str__())











