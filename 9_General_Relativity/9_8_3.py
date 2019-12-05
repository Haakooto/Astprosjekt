import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time


def V(r, M):
    return np.sqrt((1 - 2 * M / r) / r ** 2) 

def main():
    r = np.linspace(1, 50, 1000)
    M = r[0] / 2
    plt.plot(r, V(r, M))
    plt.xlabel("r")
    plt.ylabel("E/m")
    plt.show()


if __name__ == "__main__":
    main()
