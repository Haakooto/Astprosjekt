import scipy.integrate as si
import numpy as np
import matplotlib.pyplot as plt


class ExponentialDecay:
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

	def solve(self, u0, T, dt):
		n = int(T / dt)
		t = np.linspace(0, T, n)

		solved = si.solve_ivp(self, (0, T), u0, t_eval=t, method="RK45")
		return solved.t, np.transpose(solved.y)


if __name__ == "__main__":
	a = 0.0151386
	e = 0.0057
	h = 1.5520257
	ang = 0

	u0 = [0]
	T = 1
	dt = 0.000001

	decay_model = ExponentialDecay(a, e, h, ang)
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
