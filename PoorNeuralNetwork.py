# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------
import numpy as np

class WeigthInitializer:
    def initialize(self, size):
        return np.ones(size, dtype=np.float)

    def __call__(self, size):
        return self.initialize(size)


class RandomInitializer(WeigthInitializer):
    def __init__(self, shift=-0.5, scale=0.2):
        self.shift = shift
        self.scale = scale
        #self.weights = WeigthInitializer

    def initialize(self, size):
        """Random number initializer
        Note #1: 'self.scale' specifies the range of the values and with 'self.shift' they can be shifted.
        Note #2: By default (with scale=0.2 and shift=-0.5) it should return a matrix which contains random values between -0.1 and 0.1.
        Note #3: Use the np.random modul!

        :param size: Dimensions of the matrix.
        :returns: A matrix of random numbers with dimensions specified by 'size'.
        """
        # np.random.uniform(0.1, -0.1, size=size) * self.scale
        #np.random.normal(scale=self.scale, )
        return self.weights


print(RandomInitializer())

# weightes = RandomInitializer((3,3))
# print(weightes)






