import numpy as np 


def hrepeat(arr, w):
    """Flip a 1D array vertically and repeat horizontally w times."""
    return np.repeat(arr.reshape(-1, 1), repeats=w, axis=1)

def vrepeat(arr, h):
    """Flip a 1D array horizontally, and repeat vertically h times."""
    return np.repeat(arr.reshape(1, -1), repeats=h, axis=0)

def first_nonzero(arr, axis, invalid_val=-1):
    """Compute the index of the first non zero entry along an axis. If all 
    entries are zeroes, invalid_val is returned"""
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

