"""
Program for å finne og plotte planetposisjoner numerisk

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
	def __init__(self, seed, data_path=None, has_moons=True, verbose=True):
		SolarSystem.__init__(self, seed, data_path=None, has_moons=True, verbose=True)
		self.one_year = np.sqrt(self.semi_major_axes[0] ** 3 / self.star_mass)
		self.spin = self.initial_positions[0] * self.initial_velocities[1] \
			- self.initial_positions[1] * self.initial_velocities[0]

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

	def differential_orbits(self, yrs, dt_pr_yr, newpos=False):
		from ivp import Diff_eq as Deq


		if newpos:
			r0 = self.r0
		else:
			r0 = self.initial_positions

		T = self.one_year * yrs
		dt = self.one_year * dt_pr_yr

		e = self.eccentricities
		a = self.semi_major_axes
		omega = self.aphelion_angles + np.pi
		h = self.spin

		start_angle = np.arctan(r0[1] / r0[0])
		start_angle = np.where(r0[0] >= 0, start_angle, start_angle + np.pi)

		orbits = Deq(a, e, h, omega)
		t, u = orbits.solve(start_angle, T, dt)

		R = a * (1 - e ** 2) / (1 + e * np.cos(u - omega))

		x = R * np.cos(u)
		y = R * np.sin(u)

		self.d_pos = np.transpose([x, y], (0, 2, 1))

	def iterated_orbits(self, yrs, dt_pr_yr):

		def accelerate(r):
			return self.constant * r * (np.linalg.norm(r, axis=0)) ** (-3)

		T = self.one_year * yrs
		dt = self.one_year * dt_pr_yr
		nt = int(T / dt)

		self.time = np.linspace(0, T, nt)

		pos = np.zeros((2, self.number_of_planets, nt))
		pos[:, :, 0] = self.initial_positions

		self.constant = -const.G_sol * self.star_mass

		vel = self.initial_velocities

		acc_0 = accelerate(pos[:, :, 0])

		for t in range(nt - 1):
			pos[:, :, t + 1] = (
				pos[:, :, t] + vel * dt + 0.5 * acc_0 * dt ** 2
			)

			acc_1 = accelerate(pos[:, :, t + 1])
			vel = vel + 0.5 * (acc_0 + acc_1) * dt

			acc_0 = acc_1

		self.i_pos = pos


	def plot_orbits(self, a=False, i=False, d=False, o=False):

		ordered_planets = np.argsort(np.linalg.norm(self.initial_positions, axis = 0))
		planet_names = ["Vulcan", "Laconia", "Vogsphere", "Ilus", "Alderaan",
					"Apetos", "Auberon", "Zarkon", "Tellusia", "X"]

		for p in range(self.number_of_planets):
			lab = f"{planet_names[p]}; nr. {ordered_planets[p]}"
			if i:
				plt.plot(*self.i_pos[:, p, :], "b", label=f"{lab}, i")
			if d:
				plt.plot(*self.d_pos[:, p, :], "g", label=f"{lab}, d")
			if o:
				plt.plot(*self.pos[:, p, :], "c", label=f"{lab}, p")
		if a:
			plt.plot(*self.a_pos, "y", label="analytical")
		plt.scatter(*self.initial_positions, label="init")
		if i:
			plt.scatter(*self.i_pos[:,:,-1], label="final i")
		if d:
			plt.scatter(*self.d_pos[:,:,-1], label="final d")
		if o:
			plt.scatter(*self.pos[:,:,-1], label="final pos")
		plt.scatter([0], [0], s=80, c=np.array(self.star_color)/255)

		plt.grid()
		plt.axis("equal")
		plt.legend(loc=1)
		plt.show()

	def long_run(self, total_years, dt_pr_yr, batch_size=50):

		N = total_years // batch_size
		if total_years % batch_size != 0:
			total_years = N * batch_size

		T = self.one_year * batch_size

		filename = f"pos_{total_years}yr.npy"

		for n in range(N):
			if n != 0:
				prev_pos = np.load(filename)
				self.r0 = prev_pos[:, :, -1]
			else:
				self.r0 = self.initial_positions

			self.differential_orbits(batch_size, dt_pr_yr, True)

			if n != 0:
				pos = np.concatenate((prev_pos, self.d_pos), axis = 2)
			else:
				pos = self.d_pos

			np.save(filename, pos)
			self.pos = pos



if __name__ == "__main__":
	seed = util.get_seed("haakooto")

	# mission = SpaceMission(seed)
	system = SolarSys(seed)

	years = 100
	dt = 1e-4

	# system.long_run(years, dt)

	system.analytical_orbits()
	# system.iterated_orbits(years, dt)
	# system.load_pos(f"pos_{years}yr.npy")

	# system.pos = np.load("pos_100yr.npy")
	# system.pos = system.pos[:,:,::2]

	system.d_pos = np.load(f"./npys/pos_{years}yr.npy")
	# system.t = np.linspace(0, years * system.one_year, len(system.d_pos[0][0]))

	# system.differential_orbits(years, dt)

	system.plot_orbits(a=True, d=True)

	# np.save(f"pos_{years}yr", system.d_pos)

	# system.verify_planet_positions(years * system.one_year, system.d_pos)
	# system.generate_orbit_video(system.t, system.d_pos)
