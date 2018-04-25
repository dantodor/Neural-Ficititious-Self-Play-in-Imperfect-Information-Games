from __future__ import print_function
import logging
import deck


class Game:
    """Game interface which handles rules, process and players"""

    def __init__(self, player1 = None, player2 = None):
        self._players_state = [False, False, False, False]
        self._game_state = 0
        self._public_card = []
        self._pot = [0, 0]
        self._player1 = player1
        self._player2 = player2
        self._deck = deck.Deck() # Parserparameter pass to function -> size of deck

    def init_game(self):
        """Initializes game environemt. Shuffles deck, hand out cards."""
        logging.debug("Initiate Game. Shuffle Cards. Handout private Cards ...")
        self._deck.shuffle()
        self._player1.set_private_card(self._deck.pick_up())
        self._player2.set_private_card(self._deck.pick_up())
        # indice 0 -> player 1; indice 1 -> player 2

    def play_game(self):
        """Plays exactly one round -> full game."""
        logging.debug("="*45)
        logging.debug("Bettinground: 1 ...")
        self._players_state = self._play_betting_round()
        if self._players_state[2] == True:
            logging.debug("Winner is Player 2")
        elif self._players_state[3] == True:
            logging.debug("Winner ist Player 1")
        else:
            self._players_state = [False, False, False, False]
            self._public_card = self._deck.pick_up()
            logging.debug("="*45)
            logging.debug("Bettinground: 2 ...")
            self._players_state = self._play_betting_round()
            if self._players_state[2] == True:
                logging.debug("Player 2 wins.")
                # Winner is Player 2
            elif self._players_state[3] == True:
                logging.debug("Player 1 wins.")
                # Winner is Player 1
            else:
                self._evaluate_winner()

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

