import numpy as np
import sys, os
import matplotlib.pyplot as plt

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

import time

util.check_for_newer_version()

class SolarSystem(SolarSystem):
	def analytical_orbits(self):
		p = 10000
		N = self.number_of_planets

		f = np.transpose(np.array([np.linspace(0, 2 * np.pi, p)] * N))
		a = np.array([self.semi_major_axes] * p)
		e = np.array([self.eccentricities] * p)
		omega = np.array([self.aphelion_angles] * p) + np.pi

		R = a * (1 - e ** 2) / (1 + e * np.cos(f - omega))
		x = R * np.cos(f)
		y = R * np.sin(f)

		self.a_pos = np.array([x, y])

	def differential_orbits(self, T, dt):
		from ivp import ExponentialDecay as ED

		e = self.eccentricities
		a = self.semi_major_axes
		omega = self.aphelion_angles + np.pi
		r = self.initial_positions
		v = self.initial_velocities
		h = r[0] * v[1] - r[1] * v[0]

		start_angle = np.arctan(self.initial_positions[1] / self.initial_positions[0])
		start_angle = np.where(
			self.initial_positions[0] >= 0, start_angle, start_angle + np.pi
		)

		orbits = ED(a, e, h, omega)
		t, u = orbits.solve(start_angle, T, dt)

		R = a * (1 - e ** 2) / (1 + e * np.cos(u - omega))

		x = R * np.cos(u)
		y = R * np.sin(u)

		self.d_pos = np.transpose([x, y], (0, 2, 1))

	def iterated_orbits(self, T, dt):
		self.T = T
		dt = dt
		nt = int(T / dt)

		self.time = np.linspace(0, T, nt)

		pos = np.zeros((2, self.number_of_planets, nt))
		pos[:, :, 0] = self.initial_positions

		self.constant = -const.G_sol * self.star_mass

		vel = self.initial_velocities

		acc_0 = self.accelerate(pos[:, :, 0])

		for t in range(nt - 1):
			pos[:, :, t + 1] = (
				pos[:, :, t] + vel * dt + 0.5 * acc_0 * dt ** 2
			)

			acc_1 = self.accelerate(pos[:, :, t + 1])
			vel = vel + 0.5 * (acc_0 + acc_1) * dt

			acc_0 = acc_1

		self.i_pos = pos

	def accelerate(self, r):
		return self.constant * r * (np.linalg.norm(r, axis=0)) ** (-3)

	def plot_orbits(self):
		for i in range(self.number_of_planets):
			plt.plot(*self.i_pos[:, i, :], "b")
			plt.plot(*self.d_pos[:, i, :], "g")
		plt.plot(*self.a_pos, "r")

		plt.scatter(*self.i_pos[:,:,0])
		plt.scatter(*self.d_pos[:,:,0])

		plt.scatter(*self.i_pos[:,:,-1])
		plt.scatter(*self.d_pos[:,:,-1])

		plt.grid()
		plt.axis("equal")
		plt.show()

	def load_pos(self, filename):
		self.pos = np.load(filename)


if __name__ == "__main__":
	timer = time.time()

	seed = util.get_seed("haakooto")

	mission = SpaceMission(seed)
	system = SolarSystem(seed)

	system.analytical_orbits()
	year_conv = system.rotational_periods[0]
	years = 2
	dt = year_conv / 1e5

	# print(f"time to iterated orbits: {time.time()-timer}")
	system.iterated_orbits(years * year_conv, dt)
	# print(f"time to differential orbits: {time.time()-timer}")
	system.differential_orbits(years * year_conv, dt)
	# print(f"time to print: {time.time() - timer}")
	# system.load_pos("backup_20yr.npy")
	system.plot_orbits()
	# print(system.a_pos.shape)
	# print(system.i_pos.shape)
	# print(system.d_pos.shape)
	# print(system.i_pos)
	# print(system.d_pos)
	# X = system.pos
	# for i in range(system.number_of_planets):
	#     plt.plot(X[0, i, :], X[1, i, :])
	# plt.plot(system.X, system.Y)

	# plt.scatter(*system.initial_positions)
	# plt.scatter(system.X[0], system.Y[0])

	# plt.scatter(X[0, :, -1], X[1, :, -1])
	# plt.scatter(system.X[-1], system.Y[-1])

	# plt.axis("equal")
	# plt.grid()
	# plt.show()

	# np.save("planets_pos_20yr", system.pos)

	system.verify_planet_positions(years * year_conv, system.i_pos)
	system.verify_planet_positions(years * year_conv, system.d_pos)
