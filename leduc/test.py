
from __future__ import print_function
import logging
import deck
import numpy as np
import ConfigParser
import random

actions_done = []
actions_done.append('Call')
actions_done.append('Raise')

action_value = 2

class b_response():
    def predict(self, state):
        return state * 2

class avg_response():
    def predict(self, state):
        return state*2

state = 0

eta = 0.1
if random.random() > eta:
    b_response.predict(state)
else:
    avg_response.predict(state)