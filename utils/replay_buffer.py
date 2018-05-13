""" 
ReplayBuffer for reinforcement and supervised learning.

Provides:
    1. Random sample without duplicates
    2. A recent experience sample
    3. Reservoir sampling

Author: David Joos
Credits: Patrick Emami
"""
from collections import deque
import random
import numpy as np
import itertools


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)
        self.last_recent_batch = 0

    def add(self, s, a, r, s2, t):
        s = np.reshape(s, (1, 30))
        a = np.reshape(a, (1, 3))
        if s2 is not None:
            s2 = np.reshape(s2, (1, 30))
        experience = (s, a, r, s2, t)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, s2_batch, t_batch

    def recent_batch(self, batch_size):
        batch = []
        recent_window = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            recent_window = list(itertools.islice(self.buffer, self.last_recent_batch, (self.count + 1)))
            batch = random.sample(recent_window, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, s2_batch, t_batch

    def reservoir_sample(self, batch_size):
        batch = list(itertools.islice(self.buffer, 0, batch_size))
        if self.count > batch_size:
            for i in range(batch_size, (len(self.buffer) - 1)):
                j = random.randrange(1, i + 1)
                if j < batch_size:
                    batch[j] = self.buffer[i]

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1][0] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, s2_batch, t_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
