
from __future__ import print_function
import logging
import deck
import numpy as np
import ConfigParser
import random
from collections import deque
import math

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

    def test(self, lel):
        if lel:
            return True
        print("Kp")
        return False

    @property
    def stuff(self):
        return self.ones

    def print(self):
        print(self.ones)


def boltzmann(actions, temp):
    dist = np.zeros(len(actions))
    bottom = 0
    for action in actions:
        bottom += np.exp(action / temp)

    for k in range(len(actions)):
        top = np.exp(actions[k] / temp)
        dist[k] = top / bottom

    return dist

if __name__ == '__main__':

    test_q = np.array([0.0, -0.45, 0.23])
    print(test_q)
    t = 0.99
    print("--------------------")
    print(boltzmann(test_q, t))

    print((1 + 0.02 * np.sqrt(100000))**(-1))





