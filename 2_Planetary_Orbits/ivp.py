"""
Program for å løse differensiallikning d theta / d t

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
		return (
			(1 + self.e * np.cos(u - self.ang)) ** 2
			* self.h
			* (self.a * (1 - self.e ** 2)) ** (-2)
		)

	def solve(self, u0, T, dt, t0=0):
		n = int(T / dt)
		t = np.linspace(0, T, n+1)

		# use scipys integrate.solve_ivp to
		u = solve_ivp(
			self, (t0, T), u0, method="Radau", t_eval=t, atol=1e-7, rtol=1e-9
		).y

		return t, np.transpose(u)


if __name__ == "__main__":
	a = 0.0151386
	e = 0.0057
	h = 1.5520257
	ang = 0

	u0 = [0]
	T = 1
	dt = 0.000001

	decay_model = Diff_eq(a, e, h, ang)
	t, u = decay_model.solve(u0, T, dt)
	print(u[0])

	r = a * (1 - e ** 2) / (1 + e * np.cos(u - ang))

	x = r * np.cos(u)
	y = r * np.sin(u)

	plt.plot(x, y)

	plt.axis("equal")
	plt.xlabel("t")
	plt.ylabel("y")
	plt.grid()
	plt.show()
