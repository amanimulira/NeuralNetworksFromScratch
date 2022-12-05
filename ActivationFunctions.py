from fromScratch import *


def tanh(x: int):
    return np.tanh(x)

def tanh_prime(x: int):
    return 1-np.tanh(x)**2