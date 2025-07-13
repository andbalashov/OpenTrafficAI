import numpy as np

class SmoothedValue:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.deque: list[float] = []

    def update(self, value: float):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)

    @property
    def avg(self) -> float:
        return float(np.mean(self.deque)) if self.deque else 0.0