from wisard.ram import RAM

class Discriminator:
    def __init__(self, name, indexes):
        self.name = name
        self.indexes = indexes
        self.rams = [RAM() for i in range(len(self.indexes))]

    def train(self, data, bleaching=True):
        for i in range(0, len(self.indexes)):
            bitStr = ""
            for index in self.indexes[i]:
                bitStr += str(data[index])
            self.rams[i].train(bitStr, bleaching)

    def classify(self, data, bleaching=0):
        total = 0
        for i in range(0, len(self.indexes)):
            bitStr = ""
            for index in self.indexes[i]:
                bitStr += str(data[index])
            total += self.rams[i].classify(bitStr)

        return total
