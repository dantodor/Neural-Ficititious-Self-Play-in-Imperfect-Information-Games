from collections import deque
import random
import numpy as np


class ReservoirBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)
        self.last_recent_batch = 0

    def add(self, s, a):
        s = np.reshape(s, (1, 30))
        a = np.reshape(a, (1, 3))
        experience = (s, a)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            j = random.randrange(1, self.buffer_size + 1)
            if j < self.buffer_size:
                self.buffer[j] = experience

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

        return s_batch, a_batch
