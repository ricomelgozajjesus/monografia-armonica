import numpy as np

def triangular_wave(t, amplitude=1.0, frequency=1.0):
    T = 1.0 / frequency
    x = (t / T) % 1.0
    return amplitude * (4 * np.abs(x - 0.5) - 1)
