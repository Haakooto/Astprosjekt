"""
Program for sannsynlighetsmodeller

All kode er egenskrevet
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import ast2000tools.constants as const

n = int(1e5)
m = const.m_H2
T = 3000
k = const.k_B
sig_b = np.sqrt(k * T / m)


def gauss(x, sig, mu):
    return (np.sqrt(2 * np.pi) * sig) ** (-1) * np.exp(
        -0.5 * (x - mu) ** 2 * sig ** (-2)
    )


def max_boz(V, T, m):
    return (
        (m / (2 * np.pi * k * T)) ** (3 / 2)
        * np.exp(-m * V ** 2 / (2 * k * T))
        * 4
        * np.pi
        * V ** 2
    )


def max_boz_x(v, T, m):
    return np.sqrt(m / (2 * np.pi * k * T)) * np.exp(-0.5 * m * v ** 2 / (k * T))


def P_g(a, b, mu, sig):
    dx = (b - a) / n
    x = np.linspace(a, b, n)
    y = gauss(x, sig, mu)
    return x, y, np.trapz(y, dx=dx)


def P_mb(a, b, F=True, T=T, m=m):
    assert a < b, "a must be smaller than b and not the same"

    if F:
        f = max_boz
    else:
        f = max_boz_x

    dx = (b - a) / n
    x = np.linspace(a, b, n)
    y = f(x, T, m)
    return x, y, np.trapz(y, dx=dx)


def test_P_g():
    sig = 1
    mu = 0
    tol = 0.005

    s1 = P_g(-sig, sig, mu, sig)
    s2 = P_g(-2 * sig, 2 * sig, mu, sig)
    s3 = P_g(-3 * sig, 3 * sig, mu, sig)

    assert abs(0.68 - s1[2]) < tol
    assert abs(0.95 - s2[2]) < tol
    assert abs(0.997 - s3[2]) < tol

    N = np.random.normal(0, 1, n)
    N = plt.hist(N, density=True, bins="auto", label="hist of np.rd.normal")
    S = P_g(-4, 4, mu, sig)
    plt.plot(S[0], S[1], label="our function")
    plt.legend()
    plt.show()


def test_P_mb():
    N = np.random.normal(0, sig_b, n)
    m = P_mb(-15000, 15000, False)
    plt.hist(N, density=True, bins="auto")
    plt.plot(m[0], m[1])
    plt.show()


def main():
    test_P_g()
    test_P_mb()


if __name__ == "__main__":
    main()
