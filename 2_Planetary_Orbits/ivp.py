"""
Program for å løse differensiallikning d_banevinkel / d_t

All kode er egenskrevet
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class Diff_eq:
    def __init__(self, sma, ecc, spin, ang):
        self.a = sma
        self.e = ecc
        self.h = spin
        self.ang = ang

    def __call__(self, t, u):
        # equation 7 in part2, Simulating planetary orbits
        return (
            (1 + self.e * np.cos(u - self.ang)) ** 2
            * self.h
            * (self.a * (1 - self.e ** 2)) ** (-2)
        )

    def solve(self, u0, T, dt, t0=0):
        n = int(T / dt)
        t = np.linspace(0, T, n + 1)

        u = solve_ivp(
            self, (t0, T), u0, method="Radau", t_eval=t, atol=1e-7, rtol=1e-9
        ).y
        # solve_ivp returns dictionary with time points, y-values and statuses of how integration went.
        # We are here only interested in the y values, which are the orbital angles for all the planets

        return t, np.transpose(u)
