from time import time


class Stopwatch:
    def __init__(self): self.reset()
    
    def reset(self): self.start_time = time()
    
    @property
    def split(self): return time() - self.start_time