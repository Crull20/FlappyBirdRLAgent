from collections import deque
import random

# Class for storing past experiences. 
class Memory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        if seed is not None:
            random.seed(seed)

    # Adds an experience to memory
    def add(self, experience):
        self.memory.append(experience)

    # Returns a random sample of *batch_size* experiences from memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Returns # of stored experiences
    def __len__(self):
        return len(self.memory)
    
