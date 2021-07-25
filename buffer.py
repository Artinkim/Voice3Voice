import numpy as np
class CircularBuffer():
    # Initializes NumPy array and head/tail pointers
    def __init__(self, capacity, dtype=float):
        self._buffer = np.zeros(capacity, dtype)
        self._head_index = 0
        self._tail_index = 0
        self._capacity = capacity
    # Makes sure that head and tail pointers cycle back around
    def fix_indices(self):
        if self._head_index >= self._capacity:
            self._head_index -= self._capacity
            self._tail_index -= self._capacity
        elif self._head_index < 0:
            self._head_index += self._capacity
            self._tail_index += self._capacity
    # Inserts a new value in buffer, overwriting old value if full
    def insert(self, value):
        if self.is_full():
            self._head_index += 1
        self._buffer[self._tail_index % self._capacity] = value
        self._tail_index += 1
        self.fix_indices()

    # Returns the circular buffer as an array starting at head index
    def unwrap(self):
        return np.concatenate((
            self._buffer[self._head_index:min(self._tail_index, self._capacity)],
            self._buffer[:max(self._tail_index - self._capacity, 0)]
        ))
    # Indicates whether the buffer has been filled yet
    def is_full(self):
        return self.count() == self._capacity
    # Returns the amount of values currently in buffer
    def count(self):
        return self._tail_index - self._head_index
