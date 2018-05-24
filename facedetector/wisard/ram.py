import numpy as np
class RAM:
    def __init__(self):
        self.slots = {}

    def __getitem__(self, index):
        return self.slots.get(index, 0)

    def __setitem__(self, index, value):
        self.slots[index] = value

    def train(self, index, bleaching=True):
        if int(index) != 0:
            if self.slots.get(index, 0) == 0:
                self.slots[index] = 1
            elif bleaching:
                self.slots[index] += 1


    def classify(self, index, bleaching=0):
        if self.slots.get(index, 0) > bleaching:
            return 1
        else:
            return 0
        # return self.slots.get(index, 0)
