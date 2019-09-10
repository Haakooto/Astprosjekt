"""
Program for å finne planetposisjoner numerisk

All kode er egenskrevet

fil ivp.py er nødvendig for en av metodene
"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem


util.check_for_newer_version()

class SolarSys(SolarSystem):

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
		from ivp import Diff_eq as Deq

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

		orbits = Deq(a, e, h, omega)
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

	def plot_orbits(self, a=True, i=True, d=True):
		ordered_planets = [7, 1, 2, 3, 5, 6, 4]
		planet_names = ["Vulcan", "Laconia", "Vogsphere", "Ilus", "Alderaan", "Tellusia", "Auberon"]
		for p in range(self.number_of_planets):
			lab = f"{planet_names[p]}; nr. {ordered_planets[p]}"
			if i:
				plt.plot(*self.i_pos[:, p, :], "b", label=lab)
			if d:
				plt.plot(*self.d_pos[:, p, :], "g", label=lab)
		if a:
			plt.plot(*self.a_pos, "r")
		plt.scatter(*self.initial_positions, label="init")
		if i:
			plt.scatter(*self.i_pos[:,:,-1], label="final i")
		if d:
			plt.scatter(*self.d_pos[:,:,-1], label="final d")

		plt.grid()
		plt.axis("equal")
		plt.legend(loc=1)
		plt.show()

	def load_pos(self, filename):
		self.i_pos = np.load(filename)


if __name__ == "__main__":
	seed = util.get_seed("haakooto")

	mission = SpaceMission(seed)
	system = SolarSys(seed)

	one_year = 1.8556 / 20
	years = 20
	dt = one_year / 1e4

	system.analytical_orbits()
	system.iterated_orbits(years * one_year, dt)
	# system.load_pos("i_pos_20yr.npy")

	system.differential_orbits(years * one_year, dt)

	system.plot_orbits()

	# np.save("i_pos_20yr", system.i_pos)

	system.verify_planet_positions(years * one_year, system.i_pos)
	system.verify_planet_positions(years * one_year, system.d_pos)
