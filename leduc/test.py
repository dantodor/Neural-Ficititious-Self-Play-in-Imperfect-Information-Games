
from __future__ import print_function
import logging
import deck
import numpy as np
import ConfigParser

class Test:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def change(self, name):
        self._name = name

if __name__ == '__main__':
    one = Test('David')
    two = Test('Manu')

    sample = [one, two]

    print(sample[0].name)

    sample[0].change('Lukas')

    print(sample[0].name)