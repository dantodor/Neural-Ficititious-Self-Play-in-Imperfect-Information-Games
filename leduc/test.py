
from __future__ import print_function
import logging
import deck
import numpy as np
import ConfigParser

actions_done = []
actions_done.append('Call')
actions_done.append('Raise')

action_value = 2

if len(actions_done) == 2 and actions_done[0] == 'Call' and actions_done[1] == 'Raise' \
        and action_value == 2:
    action_value = 1

print (action_value)