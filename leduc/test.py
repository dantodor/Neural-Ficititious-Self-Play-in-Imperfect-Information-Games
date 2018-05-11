
from __future__ import print_function
import logging
import deck
import numpy as np
import ConfigParser
import random
from collections import deque

class env:
    def __init__(self):
        self.name = "Lol"
        self.buffer = deque()

    def set_name(self, name):
        self.name = name

    @property
    def get_name(self):
        return self.name

    def add(self, stuff):
        self.buffer.append(stuff)

    def print_buffer(self):
        print(self.buffer)

class player:
    def __init__(self, env):
        self.env = env
        self.ones = np.ones(30)

    def env_name(self):
        print(self.env.get_name)

    def change_name(self, name):
        self.env.set_name(name)

    def change_to_zeros(self):
        self.ones = np.zeros(30)

    @property
    def stuff(self):
        return self.ones

    def print(self):
        print(self.ones)




if __name__ == '__main__':
    env_ = env()
    player1 = player(env_)
    player2 = player(env_)

    env_.add(player1.stuff)
    env_.print_buffer()
    player1.print()
    player1.change_to_zeros()
    env_.print_buffer()
    player1.print()


