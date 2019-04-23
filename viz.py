import numpy as np


def symmetric_remove(x: np.ndarray, n: int) -> np.ndarray:
    """ removes n items from beginning and end """
    odd = is_odd(n)
    half = int(n / 2)
    if half > 0:
        x = x[half:-half]
    if odd:
        x = x[1:]
    return x


def is_odd(number: int) -> bool:
    """ checks whether number is odd, returns boolean  """
    return bool(number & 1)


def smooth(y, window, mode):
    """ smooth 1D vectory y    """
    return np.convolve(y, np.ones(window) / window, mode=mode)
