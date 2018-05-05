import numpy as np

class Human:

    def __init__(self, name):
        self.name = name

    def act(self):
        action = int(raw_input("Action: "))
        r_action = np.array([0, 0, 0])
        r_action[action] = 1
        return r_action

    def show_state(self, state, round):
        # print(state)
        if round == 0:
            cards = state[0][24:27]
        elif round == 1:
            cards = state[0][27:30]
        print("Round {} - Your Cards: {}".format(round, cards))

    def show_winner(self, reward, op_cards, a):
        print("="*30)
        print("Terminated:")
        print("You got reward: {}".format(reward))
        print("enemy cards: {}".format(op_cards))
        print("His last actoin: {}".format(a))
        print("="*30)
